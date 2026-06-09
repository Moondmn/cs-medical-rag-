"""
ingestion/parse_pdfs.py
───────────────────────
Merged ingestion pipeline for the breast-cancer RAG system.

What this does
──────────────
Parsing quality  → yours: OCR fallback chain, TableFormer v2, formula enrichment,
                           figure/page/table-crop PNG exports at 3× scale, GPU-aware.
RAG-readiness    → mine:  DOI / title / author / IMRaD section extraction,
                           RAG-ready JSONL schema, idempotent run with --overwrite.

Per-paper outputs (under output/parsed/<stem>/)
────────────────────────────────────────────────
  chunks.jsonl       ← RAG input: one JSON object per text / table / formula chunk
  document.md        ← full paper markdown with PICTURE placeholders
  tables.json        ← normalised table records (structured RAG / validation)
  formulas.json      ← LaTeX formula records
  pictures.json      ← figure metadata
  document.json      ← full Docling document model (image refs)
  artifacts/
    pages/           ← full-page PNGs
    pictures/        ← figure PNGs
    tables/          ← table-crop PNGs

chunks.jsonl schema (one object per line)
──────────────────────────────────────────
{
  "doc_id":          "<sha256[:16] of filename>",
  "source_file":     "smith2021.pdf",
  "title":           "Breast cancer risk ...",
  "authors":         ["Smith J", "Jones A"],
  "doi":             "10.1016/j.breast.2021.00123",  // null if not found
  "page_count":      12,
  "chunk_index":     4,
  "chunk_type":      "text" | "table" | "formula" | "figure",
  "section":         "methods",                      // canonical IMRaD label
  "section_raw":     "Materials and Methods",
  "page_start":      3,
  "text":            "...",
  // type-specific fields:
  "table_markdown":  null | "| col | ...",
  "formula_latex":   null | "\\sum_{i=1}^{n} ...",
  "figure_caption":  null | "Fig. 2 — Kaplan-Meier ...",
  "figure_type":     null | "chart" | "figure" | "diagram",
  "artifact_path":   null | "artifacts/pictures/picture-0.png"
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

# ── silence noisy Hugging Face / transformer warnings ─────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrAutoOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableStructureV2Options,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, TableItem
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import FormulaItem, PictureItem

from config import (
    OUTPUT_DIR,
    PDF_DIR,
    PIPELINE_IMAGES_SCALE,
    PIPELINE_OCR_LANG,
    PIPELINE_USE_OCR,
)
from ingest_utils import (
    build_markdown_with_placeholders,
    export_page_images,
    export_picture_images_with_metadata,
    export_table_crops,
    extract_formulas,
    extract_tables,
    normalize_markdown,
    write_formulas_json,
    write_pictures_json,
    write_tables_json,
)
from metadata import (
    canonical_section,
    extract_authors_from_docling,
    extract_doi,
    extract_title_from_docling,
)

log = logging.getLogger(__name__)


# ── GPU detection ─────────────────────────────────────────────────────────────


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


# ── OCR fallback chain (your logic, unchanged) ────────────────────────────────


def _ocr_options():
    """EasyOCR → RapidOCR → Tesseract CLI → OcrAuto."""
    try:
        import easyocr  # noqa: F401

        log.info("OCR: EasyOCR (hybrid mode, gpu=%s)", _has_cuda())
        return EasyOcrOptions(
            force_full_page_ocr=False, lang=["en"], use_gpu=_has_cuda()
        )
    except ImportError:
        pass

    try:
        import rapidocr  # noqa: F401

        log.info("OCR: RapidOCR (hybrid mode)")
        return RapidOcrOptions(
            force_full_page_ocr=False, lang=["english"], backend="onnxruntime"
        )
    except ImportError:
        pass

    if shutil.which("tesseract"):
        log.info("OCR: Tesseract CLI (lang=%s)", PIPELINE_OCR_LANG)
        return TesseractCliOcrOptions(
            force_full_page_ocr=False, lang=[PIPELINE_OCR_LANG]
        )

    log.info("OCR: OcrAuto (fallback)")
    return OcrAutoOptions(force_full_page_ocr=False)


# ── Pipeline options (your logic, unchanged) ──────────────────────────────────


def _build_pipeline_options() -> PdfPipelineOptions:
    options = PdfPipelineOptions(
        do_ocr=PIPELINE_USE_OCR,
        do_table_structure=True,
        do_formula_enrichment=True,
        images_scale=PIPELINE_IMAGES_SCALE,
        generate_page_images=True,
        generate_picture_images=True,
        ocr_options=(
            _ocr_options()
            if PIPELINE_USE_OCR
            else OcrAutoOptions(force_full_page_ocr=False)
        ),
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA if _has_cuda() else AcceleratorDevice.CPU,
            num_threads=os.cpu_count() or 4,
        ),
    )
    # TableFormer v2, cell matching off (faster, cleaner on dense clinical tables)
    options.table_structure_options = TableStructureV2Options(do_cell_matching=False)
    return options


def _build_converter() -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_build_pipeline_options())
        }
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _doc_id(filename: str) -> str:
    return hashlib.sha256(filename.encode()).hexdigest()[:16]


def _page_of(item) -> Optional[int]:
    try:
        provs = getattr(item, "prov", None)
        if provs:
            return provs[0].page_no + 1  # Docling is 0-indexed internally
    except Exception:
        pass
    return None


def _max_page(doc: DoclingDocument) -> int:
    try:
        pages = set()
        for item, _ in doc.iterate_items():
            pg = _page_of(item)
            if pg:
                pages.add(pg)
        return max(pages) if pages else 0
    except Exception:
        return 0


# ── Section tracker ───────────────────────────────────────────────────────────


class _SectionTracker:
    """Tracks current IMRaD section while walking document items in order."""

    def __init__(self) -> None:
        self.raw = "unknown"
        self.canon = "unknown"
        self.page = 1

    def update(self, heading: str, page: int) -> None:
        self.raw = heading
        self.canon = canonical_section(heading)
        self.page = page

    def as_dict(self) -> dict[str, Any]:
        return {"section": self.canon, "section_raw": self.raw}


# ── JSONL chunk builders ──────────────────────────────────────────────────────


def _text_chunk(
    base: dict[str, Any],
    idx: int,
    section: _SectionTracker,
    page: int,
    text: str,
) -> dict[str, Any]:
    return {
        **base,
        "chunk_index": idx,
        "chunk_type": "text",
        **section.as_dict(),
        "page_start": page,
        "text": text,
        "table_markdown": None,
        "formula_latex": None,
        "figure_caption": None,
        "figure_type": None,
        "artifact_path": None,
    }


def _table_chunk(
    base: dict[str, Any],
    idx: int,
    section: _SectionTracker,
    page: int,
    markdown: str,
) -> dict[str, Any]:
    return {
        **base,
        "chunk_index": idx,
        "chunk_type": "table",
        **section.as_dict(),
        "page_start": page,
        "text": markdown,  # unified text field for embedding
        "table_markdown": markdown,
        "formula_latex": None,
        "figure_caption": None,
        "figure_type": None,
        "artifact_path": None,
    }


def _formula_chunk(
    base: dict[str, Any],
    idx: int,
    section: _SectionTracker,
    page: int,
    latex: str,
    surrounding_text: str,
) -> dict[str, Any]:
    # For embedding: combine surrounding prose + LaTeX so the chunk is searchable
    text = (
        f"{surrounding_text}\nFormula: {latex}".strip()
        if surrounding_text
        else f"Formula: {latex}"
    )
    return {
        **base,
        "chunk_index": idx,
        "chunk_type": "formula",
        **section.as_dict(),
        "page_start": page,
        "text": text,
        "table_markdown": None,
        "formula_latex": latex,
        "figure_caption": None,
        "figure_type": None,
        "artifact_path": None,
    }


def _figure_chunk(
    base: dict[str, Any],
    idx: int,
    section: _SectionTracker,
    pic_meta: dict[str, Any],
) -> dict[str, Any]:
    caption = pic_meta.get("caption", "")
    text = (
        f"Figure ({pic_meta['type']}): {caption}".strip()
        if caption
        else f"Figure ({pic_meta['type']})"
    )
    return {
        **base,
        "chunk_index": idx,
        "chunk_type": "figure",
        **section.as_dict(),
        "page_start": pic_meta.get("page") or 0,
        "text": text,
        "table_markdown": None,
        "formula_latex": None,
        "figure_caption": caption,
        "figure_type": pic_meta.get("type"),
        "artifact_path": pic_meta.get("artifact"),
    }


# ── Core parse function ───────────────────────────────────────────────────────


def parse_pdf(
    pdf_path: Path, converter: DocumentConverter | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parse a single PDF.

    Returns
    -------
    chunks      : list of chunk dicts for chunks.jsonl
    publish_bag : intermediate data needed by _publish() for the other output files
    """
    if converter is None:
        converter = _build_converter()
    log.info(
        "Parsing: %s (Docling pipeline — formula enrichment can take several minutes)",
        pdf_path.name,
    )
    t0 = time.monotonic()
    result = converter.convert(pdf_path)
    log.info(
        "  Docling finished in %.1fs — status=%s",
        time.monotonic() - t0,
        result.status,
    )
    doc: DoclingDocument = result.document

    # ── Paper-level metadata ──────────────────────────────────────────────────
    full_md = doc.export_to_markdown()
    base: dict[str, Any] = {
        "doc_id": _doc_id(pdf_path.name),
        "source_file": pdf_path.name,
        "title": extract_title_from_docling(doc),
        "authors": extract_authors_from_docling(doc),
        "doi": extract_doi(full_md),
        "page_count": _max_page(doc),
    }

    # ── Walk document in reading order ────────────────────────────────────────
    chunks: list[dict[str, Any]] = []
    section = _SectionTracker()
    idx = 0
    text_buf: list[str] = []
    last_page = 1

    def flush_text_buffer() -> None:
        nonlocal idx, text_buf
        text = "\n\n".join(text_buf).strip()
        if text:
            chunks.append(_text_chunk(base, idx, section, last_page, text))
            idx += 1
        text_buf = []

    # We need picture metadata before building placeholders in markdown,
    # but we also need to walk in order. Collect picture self_refs → metadata
    # from a pre-pass so we can attach them during the main walk.
    pic_self_ref_map: dict[str, dict[str, Any]] = {}
    pic_index = 0
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            pic_self_ref_map[item.self_ref] = {
                "index": pic_index,
                "id": f"picture-{pic_index}",
                "type": "figure",  # refined later in export
                "page": _page_of(item),
                "caption": "",
                "artifact": f"artifacts/pictures/picture-{pic_index}.png",
            }
            try:
                pic_self_ref_map[item.self_ref]["caption"] = (
                    item.caption_text(doc) or ""
                )
            except Exception:
                pass
            pic_index += 1

    for item, _ in doc.iterate_items():
        label = str(getattr(item, "label", "")).lower()
        page = _page_of(item) or last_page

        # ── Section heading ───────────────────────────────────────────────────
        if "header" in label or "heading" in label or label == "title":
            flush_text_buffer()
            section.update(item.text.strip(), page)
            last_page = page
            continue

        # ── Table ─────────────────────────────────────────────────────────────
        if isinstance(item, TableItem):
            flush_text_buffer()
            try:
                md = item.export_to_markdown()
            except Exception:
                md = "[table export failed]"
            chunks.append(_table_chunk(base, idx, section, page, md))
            idx += 1
            last_page = page
            continue

        # ── Formula ───────────────────────────────────────────────────────────
        if isinstance(item, FormulaItem):
            flush_text_buffer()
            latex = item.text or item.orig or ""
            # grab nearby prose from buffer as surrounding context
            surrounding = " ".join(text_buf[-3:]) if text_buf else ""
            chunks.append(_formula_chunk(base, idx, section, page, latex, surrounding))
            idx += 1
            last_page = page
            continue

        # ── Figure / Picture ──────────────────────────────────────────────────
        if isinstance(item, PictureItem):
            flush_text_buffer()
            meta = pic_self_ref_map.get(item.self_ref, {})
            chunks.append(_figure_chunk(base, idx, section, meta))
            idx += 1
            last_page = page
            continue

        # ── Regular text block ────────────────────────────────────────────────
        text = getattr(item, "text", "").strip()
        if text:
            last_page = page
            text_buf.append(text)

    flush_text_buffer()  # flush final section

    n_by_type = {}
    for c in chunks:
        n_by_type[c["chunk_type"]] = n_by_type.get(c["chunk_type"], 0) + 1
    log.info("  → %d chunks: %s", len(chunks), n_by_type)

    publish_bag = {"doc": doc, "base": base}
    return chunks, publish_bag


# ── Secondary outputs (your _publish logic, wired to merged pipeline) ─────────


def _publish_artifacts(
    doc: DoclingDocument,
    out_dir: Path,
) -> None:
    """
    Write all non-JSONL outputs: markdown, tables.json, formulas.json,
    pictures.json, document.json, and artifact PNGs.
    This is your _publish() logic, extracted as a standalone function.
    """
    artifacts_dir = out_dir / "artifacts"

    picture_meta = export_picture_images_with_metadata(doc, artifacts_dir / "pictures")
    write_pictures_json(picture_meta, out_dir / "pictures.json")

    (out_dir / "document.md").write_text(
        normalize_markdown(build_markdown_with_placeholders(doc, picture_meta)),
        encoding="utf-8",
    )

    tables = extract_tables(doc)
    write_tables_json(tables, out_dir / "tables.json")

    formulas = extract_formulas(doc)
    write_formulas_json(formulas, out_dir / "formulas.json")

    doc.save_as_json(
        out_dir / "document.json",
        artifacts_dir=artifacts_dir,
        image_mode=ImageRefMode.REFERENCED,
    )

    n_pages = export_page_images(doc, artifacts_dir / "pages")
    n_crops = export_table_crops(doc, artifacts_dir / "tables")
    log.info(
        "  → Artifacts: tables=%d, formulas=%d, pictures=%d, pages=%d, table_crops=%d",
        len(tables),
        len(formulas),
        len(picture_meta),
        n_pages,
        n_crops,
    )


# ── JSONL writer ──────────────────────────────────────────────────────────────


def _write_jsonl(chunks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("  → JSONL: %s (%d lines)", path, len(chunks))


# ── Pipeline entry point ──────────────────────────────────────────────────────


def run_ingestion(
    pdf_dir: Path = PDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    overwrite: bool = False,
) -> None:
    """
    Parse all PDFs in *pdf_dir*.

    Each paper gets its own subdirectory under *output_dir*:
      output_dir/<stem>/chunks.jsonl   ← RAG input
      output_dir/<stem>/document.md
      output_dir/<stem>/tables.json
      ... (see module docstring for full list)
    """
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        log.warning("No PDFs found in %s — drop files there and re-run.", pdf_dir)
        return

    log.info("Found %d PDF(s) in %s", len(pdf_paths), pdf_dir)
    ok = skipped = failed = 0

    for pdf_path in pdf_paths:
        paper_out = output_dir / pdf_path.stem
        jsonl_path = paper_out / "chunks.jsonl"

        if jsonl_path.exists() and not overwrite:
            log.info("  Skipping (already parsed): %s", pdf_path.name)
            skipped += 1
            continue

        try:
            # Fresh converter per PDF (matches docling-medical-a1.py).
            chunks, bag = parse_pdf(pdf_path)
            paper_out.mkdir(parents=True, exist_ok=True)
            _write_jsonl(chunks, jsonl_path)
            _publish_artifacts(bag["doc"], paper_out)
            ok += 1
        except Exception as exc:
            log.error("  FAILED %s: %s", pdf_path.name, exc, exc_info=True)
            failed += 1

    log.info(
        "\nIngestion complete — %d parsed, %d skipped, %d failed.", ok, skipped, failed
    )

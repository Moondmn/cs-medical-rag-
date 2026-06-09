"""
Docling ingest for medical papers — RAG-ready parse, no chunking.

Ingests every PDF in papers/test_ingest/ and writes under output/ingest/<stem>/:
  document.md, tables.json, formulas.json, pictures.json, document.json, artifacts/

Pipeline (fixed):
  OCR: EasyOCR (preferred) -> RapidOCR -> Tesseract CLI -> OcrAuto
  TableFormer v2, cell matching, formula enrichment, images_scale=3.0
  Page + picture PNG export, table crops, custom PICTURE placeholder tags in markdown

Prefetch once online: docling-tools models download (HF_TOKEN optional).

Run:
  uv run python playground/ingest/docling-medical-a1.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import warnings

from pathlib import Path

_INGEST_DIR = Path(__file__).resolve().parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

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
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.base import ImageRefMode

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

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "papers" / "test_ingest"
OUT_DIR = PROJECT_ROOT / "output" / "ingest"

# --- pipeline ---
IMAGES_SCALE = 3.0
OCR_LANG = "eng"

_log = logging.getLogger(__name__)


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _ocr_options() -> (
    EasyOcrOptions | RapidOcrOptions | TesseractCliOcrOptions | OcrAutoOptions
):
    """EasyOCR -> RapidOCR -> Tesseract CLI -> OcrAuto (hybrid OCR, not full-page)."""
    try:
        import easyocr  # noqa: F401

        _log.info("Using EasyOCR (lang=en, hybrid mode).")
        return EasyOcrOptions(
            force_full_page_ocr=False,
            lang=["en"],
            use_gpu=_has_cuda(),
        )
    except ImportError:
        pass

    try:
        import rapidocr  # noqa: F401

        _log.info("Using RapidOCR (lang=english, hybrid mode).")
        return RapidOcrOptions(
            force_full_page_ocr=False,
            lang=["english"],
            backend="onnxruntime",
        )
    except ImportError:
        pass

    if shutil.which("tesseract"):
        _log.info("Using Tesseract CLI (lang=%s, hybrid mode).", OCR_LANG)
        return TesseractCliOcrOptions(force_full_page_ocr=False, lang=[OCR_LANG])

    _log.info("No EasyOCR/RapidOCR/Tesseract; using OcrAutoOptions for hybrid OCR.")
    return OcrAutoOptions(force_full_page_ocr=False)


def _pipeline_options() -> PdfPipelineOptions:
    options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        do_formula_enrichment=True,
        images_scale=IMAGES_SCALE,
        generate_page_images=True,
        generate_picture_images=True,
        ocr_options=_ocr_options(),
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA if _has_cuda() else AcceleratorDevice.CPU,
            num_threads=os.cpu_count() or 4,
        ),
    )
    options.table_structure_options = TableStructureV2Options(do_cell_matching=False)
    return options


def _publish(doc: DoclingDocument, *, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
    n_pics = len(picture_meta)
    n_crops = export_table_crops(doc, artifacts_dir / "tables")

    _log.info(
        "Wrote %s (tables=%d, formulas=%d, pictures=%d, pages=%d, table_crops=%d)",
        out_dir,
        len(tables),
        len(formulas),
        n_pics,
        n_pages,
        n_crops,
    )


def ingest_pdf(pdf_path: Path) -> None:
    stem = pdf_path.stem
    out_dir = OUT_DIR / stem
    _log.info("Converting %s", pdf_path.name)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options()),
        }
    )
    doc = converter.convert(pdf_path).document
    _publish(doc, stem=stem, out_dir=out_dir)

    (PROJECT_ROOT / "output.md").write_text(
        (out_dir / "document.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        _log.error("No PDFs in %s", PDF_DIR)
        return 1

    failures = 0
    for pdf_path in pdfs:
        try:
            ingest_pdf(pdf_path)
        except Exception:
            failures += 1
            _log.exception("Failed %s", pdf_path.name)
        break  # just for testing (one at a time)

    if failures:
        _log.error("%d of %d PDF(s) failed", failures, len(pdfs))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import json
import logging
import os
import sys

from pathlib import Path


_INGEST_DIR = Path(__file__).resolve().parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import (
    ConfidenceReport,
    ConversionStatus,
    InputFormat,
    QualityGrade,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmEngineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument
from docling.datamodel.pipeline_options import smolvlm_picture_description

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "papers" / "test_ingest"
OUT_DIR = PROJECT_ROOT / "output" / "ingest"

_log = logging.getLogger(__name__)


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _pipeline_options() -> PdfPipelineOptions:
    options = PdfPipelineOptions(
        do_picture_description=True,
        # picture_description_options=PictureDescriptionVlmEngineOptions.from_preset(
        #     "smolvlm"
        # ),
        do_ocr=True,
        do_table_structure=True,
        do_formula_enrichment=True,
        images_scale=2.0,
        generate_page_images=True,
        generate_picture_images=True,
        # ocr_options=_ocr_options(),
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA if _has_cuda() else AcceleratorDevice.CPU,
            num_threads=os.cpu_count() or 4,
        ),
    )
    options.picture_description_options = (
        smolvlm_picture_description  # <-- the model choice
    )
    options.picture_description_options.prompt = (
        "Describe the image in three sentences. Be consise and accurate."
    )
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = True
    return options


# def _publish(doc: DoclingDocument, *, stem: str, out_dir: Path) -> None:
#     out_dir.mkdir(parents=True, exist_ok=True)
#     artifacts_dir = out_dir / "artifacts"

#     picture_meta = export_picture_images_with_metadata(doc, artifacts_dir / "pictures")
#     write_pictures_json(picture_meta, out_dir / "pictures.json")

#     (out_dir / "document.md").write_text(
#         normalize_markdown(build_markdown_with_placeholders(doc, picture_meta)),
#         encoding="utf-8",
#     )

#     tables = extract_tables(doc)
#     write_tables_json(tables, out_dir / "tables.json")

#     formulas = extract_formulas(doc)
#     write_formulas_json(formulas, out_dir / "formulas.json")

#     doc.save_as_json(
#         out_dir / "document.json",
#         artifacts_dir=artifacts_dir,
#         image_mode=ImageRefMode.REFERENCED,
#     )

#     n_pages = export_page_images(doc, artifacts_dir / "pages")
#     n_pics = len(picture_meta)
#     n_crops = export_table_crops(doc, artifacts_dir / "tables")

#     _log.info(
#         "Wrote %s (tables=%d, formulas=%d, pictures=%d, pages=%d, table_crops=%d)",
#         out_dir,
#         len(tables),
#         len(formulas),
#         n_pics,
#         n_pages,
#         n_crops,
#     )

from ingest_utils import (
    build_markdown_with_placeholders,
    export_picture_images_with_metadata,
    normalize_markdown,
    write_pictures_json,
)

# Grades that suggest manual review or an alternate pipeline (Docling confidence docs).
_REVIEW_GRADES = frozenset({QualityGrade.POOR, QualityGrade.FAIR})


def write_confidence_json(confidence: ConfidenceReport, path: Path) -> None:
    """Persist document- and page-level confidence (scores + grades)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(confidence.model_dump_json(indent=2), encoding="utf-8")


def confidence_summary(confidence: ConfidenceReport) -> dict:
    """Compact summary — focus on mean_grade / low_grade per Docling guidance."""
    return {
        "mean_grade": confidence.mean_grade.value,
        "low_grade": confidence.low_grade.value,
        "pages": [
            {
                "page": page_no,
                "mean_grade": page_conf.mean_grade.value,
                "low_grade": page_conf.low_grade.value,
            }
            for page_no, page_conf in sorted(confidence.pages.items())
        ],
    }


def log_confidence(pdf_name: str, confidence: ConfidenceReport) -> None:
    """Log document-level grades; warn when review may be needed."""
    _log.info(
        "%s confidence: mean_grade=%s low_grade=%s (%d page(s))",
        pdf_name,
        confidence.mean_grade.value,
        confidence.low_grade.value,
        len(confidence.pages),
    )
    for page_no, page_conf in sorted(confidence.pages.items()):
        if page_conf.low_grade in _REVIEW_GRADES:
            _log.warning(
                "%s page %s low_grade=%s — consider manual review",
                pdf_name,
                page_no,
                page_conf.low_grade.value,
            )
    if confidence.low_grade in _REVIEW_GRADES:
        _log.warning(
            "%s document low_grade=%s — conversion may need attention",
            pdf_name,
            confidence.low_grade.value,
        )


def _publish(doc: DoclingDocument, *, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"

    picture_meta = export_picture_images_with_metadata(doc, artifacts_dir / "pictures")
    write_pictures_json(picture_meta, out_dir / "pictures.json")

    (out_dir / "document.md").write_text(
        normalize_markdown(build_markdown_with_placeholders(doc, picture_meta)),
        encoding="utf-8",
    )


def ingest_pdf(pdf_path: Path) -> None:
    stem = pdf_path.stem
    out_dir = OUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Converting %s", pdf_path.name)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options()),
        }
    )
    result: ConversionResult = converter.convert(pdf_path, page_range=(5, 7))

    write_confidence_json(result.confidence, out_dir / "confidence.json")
    (out_dir / "confidence_summary.json").write_text(
        json.dumps(confidence_summary(result.confidence), indent=2),
        encoding="utf-8",
    )
    log_confidence(pdf_path.name, result.confidence)

    if result.status != ConversionStatus.SUCCESS:
        _log.warning(
            "Conversion status %s for %s",
            result.status.value,
            pdf_path.name,
        )

    doc = result.document
    _publish(doc, stem=stem, out_dir=out_dir)
    with open(out_dir / "document.json", "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(mode="json"), f, indent=2)

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
        break  # break on first one

    if failures:
        _log.error("%d of %d PDF(s) failed", failures, len(pdfs))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

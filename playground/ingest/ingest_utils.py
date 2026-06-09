"""Helpers for Docling PDF ingest: tables, formulas, pictures, and image crops."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from docling_core.types.doc import DoclingDocument, DocItem, TableItem
from docling_core.types.doc.document import FormulaItem, PictureItem, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

# e.g. "1 . 14" -> "1.14"
_DECIMAL_SPACES = re.compile(r"(\d)\s+\.\s+(\d)")
# e.g. "2 ∗ ( height" -> "2 * ( height"
_STAR_SPACES = re.compile(r"\s*[∗×]\s*")
# TableFormer sometimes drops the leading "1." on height formula cells.
_HEIGHT_FORMULA = re.compile(
    r"^0?5\s*\+\s*2\s*\*\s*\(\s*height\s*-\s*1\s*\.?\s*$",
    re.IGNORECASE,
)

_DEFAULT_IMAGE_PLACEHOLDER = "<!-- image -->"


def normalize_markdown(text: str) -> str:
    """Fix common PDF text-layer spacing in exported markdown."""
    text = _DECIMAL_SPACES.sub(r"\1.\2", text)
    return _STAR_SPACES.sub("*", text)


def normalize_cell_text(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if not isinstance(value, str):
        return value
    text = _DECIMAL_SPACES.sub(r"\1.\2", value.strip())
    text = _STAR_SPACES.sub("*", text)
    if _HEIGHT_FORMULA.match(text):
        return "1.05 + 2*(height-1.6)"
    return text


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(normalize_cell_text)


def _label_column(df: pd.DataFrame) -> Any | None:
    """First column that carries row-span labels (merged cells in the PDF)."""
    if df.empty:
        return None
    col = df.columns[0]
    name = str(col)
    if name in {"0", "Term", "Category"} or not name.isdigit():
        return col
    return None


def fill_merged_row_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill blank cells in the label column.

    Docling often emits one row per visual line; PDF row-spans leave the first
    column empty on continuation rows, which breaks grouping and misaligns values.
    """
    label_col = _label_column(df)
    if label_col is None:
        return df

    out = df.copy()
    series = out[label_col].map(
        lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
    )
    series = series.mask(series == "", pd.NA).ffill().fillna("")
    out[label_col] = series
    return out


def extract_tables(doc: DoclingDocument) -> list[dict[str, Any]]:
    """Export all tables as normalized records (for structured RAG / validation)."""
    tables: list[dict[str, Any]] = []
    for item, _level in doc.iterate_items():
        if not isinstance(item, TableItem):
            continue
        df = fill_merged_row_labels(normalize_dataframe(item.export_to_dataframe(doc=doc)))
        record: dict[str, Any] = {
            "index": len(tables),
            "rows": len(df),
            "columns": [str(c) for c in df.columns.tolist()],
            "data": json.loads(df.to_json(orient="records")),
            "markdown": df.to_markdown(index=False),
        }
        try:
            record["html"] = item.export_to_html(doc=doc)
        except Exception:
            pass
        tables.append(record)
    return tables


def write_tables_json(tables: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")


def write_formulas_json(formulas: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(formulas, indent=2, ensure_ascii=False), encoding="utf-8")


def write_pictures_json(pictures: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pictures, indent=2, ensure_ascii=False), encoding="utf-8")


def _bbox_top_left(doc: DoclingDocument, prov: ProvenanceItem) -> list[float] | None:
    page = doc.pages.get(prov.page_no)
    if page is None or page.size is None:
        return None
    return list(prov.bbox.to_top_left_origin(page_height=page.size.height).as_tuple())


def _picture_type(item: PictureItem) -> str:
    label = item.label
    val = label.value if hasattr(label, "value") else str(label)
    if val == DocItemLabel.CHART.value or val.endswith("_chart"):
        return "chart"
    if "diagram" in val:
        return "diagram"
    if val == DocItemLabel.PICTURE.value:
        return "figure"
    return "unknown"


def _format_picture_placeholder(
    *,
    picture_id: str,
    page: int | None,
    pic_type: str,
    bbox: list[float] | None,
    caption: str,
    artifact: str,
) -> str:
    cap = caption.replace('"', "'").replace("\n", " ").strip()
    if bbox:
        bbox_str = "[" + ",".join(f"{v:.2f}" for v in bbox) + "]"
    else:
        bbox_str = "[]"
    page_str = str(page) if page is not None else "?"
    return (
        f'<!-- PICTURE[id={picture_id}, page={page_str}, type={pic_type}, '
        f"bbox={bbox_str}, caption=\"{cap}\", artifact=\"{artifact}\"] -->"
    )


def export_picture_images_with_metadata(
    doc: DoclingDocument,
    pictures_dir: Path,
) -> list[dict[str, Any]]:
    """Crop picture PNGs and return metadata records for pictures.json."""
    pictures_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for item, _level in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        index = len(records)
        picture_id = f"picture-{index}"
        artifact_name = f"{picture_id}.png"
        artifact_rel = f"artifacts/pictures/{artifact_name}"

        page: int | None = None
        bbox: list[float] | None = None
        if item.prov:
            prov = item.prov[0]
            page = prov.page_no + 1
            bbox = _bbox_top_left(doc, prov)

        caption = ""
        try:
            caption = item.caption_text(doc) or ""
        except Exception:
            pass

        image = item.get_image(doc)
        if image is not None:
            image.save(pictures_dir / artifact_name)

        pic_type = _picture_type(item)
        placeholder = _format_picture_placeholder(
            picture_id=picture_id,
            page=page,
            pic_type=pic_type,
            bbox=bbox,
            caption=caption,
            artifact=artifact_rel,
        )

        records.append(
            {
                "index": index,
                "id": picture_id,
                "type": pic_type,
                "page": page,
                "bbox": bbox,
                "caption": caption,
                "artifact": artifact_rel,
                "self_ref": item.self_ref,
                "placeholder": placeholder,
            }
        )

    return records


def build_markdown_with_placeholders(
    doc: DoclingDocument,
    picture_meta: list[dict[str, Any]],
) -> str:
    """Export markdown and replace default image placeholders with custom PICTURE tags."""
    md = doc.export_to_markdown(image_placeholder=_DEFAULT_IMAGE_PLACEHOLDER)
    for record in picture_meta:
        md = md.replace(_DEFAULT_IMAGE_PLACEHOLDER, record["placeholder"], 1)
    return md


def export_page_images(doc: DoclingDocument, pages_dir: Path) -> int:
    """Write full-page PNGs (requires generate_page_images in pipeline options)."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for page_no in sorted(doc.pages.keys()):
        page = doc.pages[page_no]
        if page.image is None:
            continue
        pil = page.image.pil_image
        if pil is None:
            continue
        pil.save(pages_dir / f"page-{page_no:03d}.png")
        count += 1
    return count


def export_table_crops(doc: DoclingDocument, artifacts_dir: Path) -> int:
    """
    Write table PNG crops via TableItem.get_image (requires generate_page_images).

    Docling guidance: prefer page images + get_image over deprecated generate_table_images.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for item, _level in doc.iterate_items():
        if not isinstance(item, TableItem):
            continue
        image = item.get_image(doc)
        if image is None:
            continue
        out_path = artifacts_dir / f"table-{count}.png"
        image.save(out_path)
        count += 1
    return count


def extract_formulas(doc: DoclingDocument) -> list[dict[str, Any]]:
    """Export formula items with LaTeX text and provenance."""
    formulas: list[dict[str, Any]] = []
    for item, _level in doc.iterate_items():
        if not isinstance(item, FormulaItem):
            continue
        record: dict[str, Any] = {
            "index": len(formulas),
            "text": item.text,
            "orig": item.orig,
            "self_ref": item.self_ref,
        }
        if item.prov:
            prov = item.prov[0]
            record["page"] = prov.page_no + 1
            bbox = _bbox_top_left(doc, prov)
            if bbox is not None:
                record["bbox"] = bbox
        formulas.append(record)
    return formulas

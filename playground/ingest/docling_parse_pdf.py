"""Per-page PDF parsing via Docling (PaperQA ``parse_pdf_to_pages`` port)."""

from __future__ import annotations

import collections
import io
import json
import logging
import os
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

_INGEST_DIR = Path(__file__).resolve().parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from ingest_utils import (  # noqa: E402
    write_formulas_json,
    write_pictures_json,
    write_tables_json,
)

import docling
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.exceptions import ConversionError
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import (
    DescriptionAnnotation,
    DocItem,
    FormulaItem,
    PictureItem,
    TableItem,
    TextItem,
)

try:
    from paperqa.types import ParsedMedia, ParsedMetadata, ParsedText
    from paperqa.utils import ImpossibleParsingError
except ImportError:
    from parsed_types import (  # type: ignore[no-redef]
        ImpossibleParsingError,
        ParsedMedia,
        ParsedMetadata,
        ParsedText,
    )

if TYPE_CHECKING:
    from docling.backend.abstract_backend import AbstractDocumentBackend

DOCLING_VERSION = version(docling.__name__)
DOCLING_IMAGES_SCALE_PER_DPI = (
    72  # SEE: https://github.com/docling-project/docling/issues/2405
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "ingest"

_log = logging.getLogger(__name__)


def parse_pdf_to_pages(  # noqa: PLR0912
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: int | tuple[int, int] | None = None,
    parse_media: bool = True,
    pipeline_cls: type = StandardPdfPipeline,
    dpi: int | None = None,
    custom_pipeline_options: Mapping[str, Any] | None = None,
    backend: "type[AbstractDocumentBackend]" = DoclingParseDocumentBackend,
    **_,
) -> ParsedText:
    """Parse a PDF into per-page text and optional media.

    Args:
        path: Path to the PDF file to parse.
        page_size_limit: Sensible character limit one page's text,
            used to catch bad PDF reads.
        parse_media: Flag to also parse media (e.g. images, tables).
        pipeline_cls: Optional custom pipeline class for document conversion.
            Default is Docling's standard PDF pipeline.
        dpi: Optional DPI (dots per inch) for image resolution,
            if left unspecified Docling's default 1.0 scale will be employed.
        custom_pipeline_options: Optional keyword arguments to use to construct the
            PDF pipeline's options.
        page_range: Optional start_page or two-tuple of inclusive (start_page, end_page)
            to parse only specific pages, where pages are one-indexed.
            Leaving as the default of None will parse all pages.
        backend: PDF backend class to use for parsing, defaults to docling-parse.
        **_: Thrown away kwargs.
    """
    path = Path(path)

    if parse_media:
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            generate_table_images=True,
            images_scale=1.0 if dpi is None else dpi / DOCLING_IMAGES_SCALE_PER_DPI,
            **(custom_pipeline_options or {}),
        )
    else:
        pipeline_options = PdfPipelineOptions(**(custom_pipeline_options or {}))

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=pipeline_cls,
                backend=backend,
            )
        }
    )
    try:
        # NOTE: this conversion is synchronous, because many backends only support sync
        # https://github.com/docling-project/docling/issues/2229#issuecomment-3269019929
        result = converter.convert(
            path,
            page_range=(
                (page_range, page_range)
                if isinstance(page_range, int)
                else (page_range or DEFAULT_PAGE_RANGE)
            ),
        )
    except ConversionError as exc:
        raise ImpossibleParsingError(
            f"PDF reading via {docling.__name__} failed on the PDF at path {path!r},"
            " likely this PDF file is corrupt."
        ) from exc
    if result.status != ConversionStatus.SUCCESS:
        raise ImpossibleParsingError(
            f"Docling conversion failed with status {result.status.value!r}"
            f" for the PDF at path {path!r}."
        )

    doc = result.document

    # NOTE: the list value here is a two-item list of page text, page media.
    # It's mutable so we can append text and media as found
    content: dict[str, list] = collections.defaultdict(lambda: ["", []])
    total_length = count_media = 0

    for item, __ in doc.iterate_items():
        if not isinstance(item, DocItem) or not item.prov:
            raise NotImplementedError(
                f"Didn't yet handle the shape of node item {item}."
            )

        # NOTE: docling pages are 1-indexed
        page_nums = [prov.page_no for prov in item.prov]

        if isinstance(item, TextItem | FormulaItem):  # Handle items with text
            item_text = item.text
            if not item_text and isinstance(item, FormulaItem) and item.orig:
                # Sometimes the sanitization of formula text fails, so use the original
                item_text = item.orig
            for page_num in page_nums:
                new_text = (
                    item_text if not content[str(page_num)][0] else "\n\n" + item_text
                )
                total_length += len(new_text)
                if page_size_limit and total_length > page_size_limit:
                    raise ImpossibleParsingError(
                        f"The text in page {page_num} was {total_length} chars long,"
                        f" which exceeds the {page_size_limit} char limit"
                        f" for the PDF at path {path}."
                    )
                content[str(page_num)][0] += new_text

        if parse_media and isinstance(  # Handle images and formulae
            item, PictureItem | FormulaItem
        ):
            image_data = item.get_image(doc)
            if image_data:
                try:
                    (page_num,) = page_nums
                except ValueError as exc:
                    raise NotImplementedError(
                        f"Picture item spanning multiple pages {page_nums}"
                        " is not yet handled."
                    ) from exc

                # Convert PIL Image to bytes (PNG format)
                img_bytes = io.BytesIO()
                image_data.save(img_bytes, format="PNG")
                img_bytes.seek(0)  # Reset pointer before read to avoid empty data

                media_metadata = {
                    "type": "formula" if isinstance(item, FormulaItem) else "picture",
                    "width": image_data.width,
                    "height": image_data.height,
                    "bbox": item.prov[0].bbox.as_tuple(),
                    "images_scale": pipeline_options.images_scale,
                }
                annotations = [
                    x
                    for x in getattr(item, "annotations", [])
                    if isinstance(x, DescriptionAnnotation)
                ]
                if len(annotations) == 1:
                    # We don't set this text in ParsedMedia.text because it's
                    # a synthetic description, not actually text in the PDF,
                    # and we don't want citations going to synthetic text
                    media_metadata.update(
                        {
                            "description_text": annotations[0].text,
                            "description_provenance": annotations[0].provenance,
                        }
                    )
                elif len(annotations) > 1:
                    raise NotImplementedError(
                        f"Didn't yet handle 2+ picture description annotations {annotations}."
                    )

                media_metadata["info_hashable"] = json.dumps(
                    {
                        k: (
                            v
                            if k != "bbox"
                            # Enables bbox deduplication based on whole pixels,
                            # since <1-px differences are just noise
                            else tuple(round(x) for x in cast(tuple, v))
                        )
                        for k, v in media_metadata.items()
                    },
                    sort_keys=True,
                )
                # Add page number after info_hashable so differing pages
                # don't break the cache key
                media_metadata["page_num"] = page_num
                content[str(page_num)][1].append(
                    ParsedMedia(
                        index=len(content[str(page_num)][1]),
                        data=img_bytes.read(),
                        info=media_metadata,
                    )
                )
                count_media += 1

        elif parse_media and isinstance(item, TableItem):  # Handle tables
            table_image_data = item.get_image(doc)
            if table_image_data:
                try:
                    (page_num,) = page_nums
                except ValueError as exc:
                    raise NotImplementedError(
                        f"Table item spanning multiple pages {page_nums}"
                        " is not yet handled."
                    ) from exc

                img_bytes = io.BytesIO()
                table_image_data.save(img_bytes, format="PNG")
                img_bytes.seek(0)  # Reset pointer before read to avoid empty data

                media_metadata = {
                    "type": "table",
                    "width": table_image_data.width,
                    "height": table_image_data.height,
                    "bbox": item.prov[0].bbox.as_tuple(),
                    "images_scale": pipeline_options.images_scale,
                }
                media_metadata["info_hashable"] = json.dumps(
                    {
                        k: (
                            v
                            if k != "bbox"
                            # Enables bbox deduplication based on whole pixels,
                            # since <1-px differences are just noise
                            else tuple(round(x) for x in cast(tuple, v))
                        )
                        for k, v in media_metadata.items()
                    },
                    sort_keys=True,
                )
                # Add page number after info_hashable so differing pages
                # don't break the cache key
                media_metadata["page_num"] = page_num
                content[str(page_num)][1].append(
                    ParsedMedia(
                        index=len(content[str(page_num)][1]),
                        data=img_bytes.read(),
                        text=item.export_to_markdown(doc),
                        info=media_metadata,
                    )
                )
                count_media += 1

    multimodal_string = f"|multimodal|images_scale={pipeline_options.images_scale}" + (
        "" if not custom_pipeline_options else f"|options={custom_pipeline_options}"
    )
    metadata = ParsedMetadata(
        parsing_libraries=[f"{docling.__name__} ({DOCLING_VERSION})"],
        total_parsed_text_length=total_length,
        count_parsed_media=count_media,
        name=(
            f"pdf|pipeline={pipeline_cls.__name__}"
            f"|page_range={str(page_range).replace(' ', '')}"  # Remove space in tuple
            f"|backend={backend.__name__}"
            f"{multimodal_string if parse_media else ''}"
        ),
    )
    return ParsedText(
        # Convert content from list to 2-tuple for return
        content={
            pgn: text if not parse_media else (text, images)
            for pgn, (text, images) in sorted(content.items(), key=lambda x: int(x[0]))
        },
        metadata=metadata,
    )


def _media_record(
    media: ParsedMedia,
    *,
    page: int,
    artifact_rel: str,
) -> dict[str, Any]:
    info = {k: v for k, v in media.info.items() if k != "info_hashable"}
    record: dict[str, Any] = {
        "index": media.index,
        "page": page,
        "type": info.get("type", "unknown"),
        "artifact": artifact_rel,
        "info": info,
    }
    if media.text:
        record["text"] = media.text
    if description := info.get("description_text"):
        record["description"] = description
    return record


def publish_parsed_text(parsed: ParsedText, *, out_dir: Path) -> None:
    """Write per-page parse under output/ingest/<stem>/ (same layout as other ingests)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_media = out_dir / "artifacts" / "media"
    artifacts_pages = out_dir / "artifacts" / "pages"
    artifacts_media.mkdir(parents=True, exist_ok=True)
    artifacts_pages.mkdir(parents=True, exist_ok=True)

    pictures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    page_records: list[dict[str, Any]] = []
    md_parts: list[str] = []

    content = parsed.content
    n_pages = 1
    if isinstance(content, str):
        (out_dir / "document.md").write_text(content, encoding="utf-8")
        (out_dir / "pages.json").write_text(
            json.dumps([{"page": 1, "text": content, "media": []}], indent=2),
            encoding="utf-8",
        )
    else:
        for page_key in sorted(content.keys(), key=int):
            page_no = int(page_key)
            if isinstance(content[page_key], tuple):
                text, media_list = content[page_key]
            else:
                text, media_list = content[page_key], []

            (artifacts_pages / f"page-{page_no:03d}.txt").write_text(
                text, encoding="utf-8"
            )
            md_parts.append(f"<!-- page {page_no} -->\n\n{text}")

            page_media_records: list[dict[str, Any]] = []
            for media in media_list:
                media_type = str(media.info.get("type", "unknown"))
                artifact_rel = (
                    f"artifacts/media/page-{page_no:03d}-"
                    f"{media.index:03d}-{media_type}.png"
                )
                (out_dir / artifact_rel).write_bytes(media.data)
                record = _media_record(media, page=page_no, artifact_rel=artifact_rel)
                page_media_records.append(record)

                if media_type == "picture":
                    pictures.append(record)
                elif media_type == "table":
                    tables.append(record)
                elif media_type == "formula":
                    formulas.append(record)

            page_records.append(
                {"page": page_no, "text": text, "media": page_media_records}
            )

        n_pages = len(page_records)
        (out_dir / "document.md").write_text("\n\n".join(md_parts), encoding="utf-8")
        (out_dir / "pages.json").write_text(
            json.dumps(page_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    meta = parsed.metadata
    if hasattr(meta, "model_dump"):
        meta_dict = meta.model_dump()
    else:
        meta_dict = {
            "parsing_libraries": meta.parsing_libraries,
            "total_parsed_text_length": meta.total_parsed_text_length,
            "count_parsed_media": meta.count_parsed_media,
            "name": meta.name,
        }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_pictures_json(pictures, out_dir / "pictures.json")
    write_tables_json(tables, out_dir / "tables.json")
    write_formulas_json(formulas, out_dir / "formulas.json")

    _log.info(
        "Wrote %s (pages=%d, pictures=%d, tables=%d, formulas=%d)",
        out_dir,
        n_pages,
        len(pictures),
        len(tables),
        len(formulas),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pdf_path = (PROJECT_ROOT / "papers" / "test_ingest" / "19-STS729.pdf").resolve()
    parsed = parse_pdf_to_pages(
        pdf_path,
        page_range=(1, 3),
        parse_media=True,
        dpi=216,
    )
    out_dir = OUT_DIR / pdf_path.stem
    publish_parsed_text(parsed, out_dir=out_dir)
    (PROJECT_ROOT / "output.md").write_text(
        (out_dir / "document.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

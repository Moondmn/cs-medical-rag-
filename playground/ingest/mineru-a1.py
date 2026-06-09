"""
Ingest medical PDFs with MinerU (lmdeploy-accelerated VLM) for breast-cancer RAG.

Uses the hybrid + lmdeploy backend (not pipeline). On Linux, *-auto-engine
backends prefer vllm when both are installed; this script pins hybrid-lmdeploy-engine
so inference runs through lmdeploy as documented:
https://opendatalab.github.io/MinerU/quick_start/extension_modules/

Install:
  uv sync
  # or: uv pip install "mineru[core,lmdeploy]"

Requires NVIDIA GPU (Volta+, 8GB+ VRAM recommended).

First run downloads MinerU weights (~2.3GB VLM + pipeline models). This is not a hang.
Use --download-models-only to prefetch, then ingest. Partial downloads resume automatically.
Set HF_TOKEN for faster Hugging Face pulls; or use --model-source modelscope in some regions.

Outputs per paper under output/ingest/<pdf_stem>/:
  document.md       — normalized markdown for chunking
  tables.json       — structured tables from MinerU content_list
  chunks.json       — RAG-ready chunks with metadata (page, section, type)
  content_list.json — MinerU reading-order blocks (passthrough)
  mineru/           — raw MinerU parse dir (hybrid_auto/)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

_INGEST_DIR = Path(__file__).resolve().parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

from ingest_utils import normalize_markdown, write_tables_json  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_DIR = PROJECT_ROOT / "papers" / "test_ingest"
DEFAULT_PDF = DEFAULT_PDF_DIR / "19-STS729.pdf"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "ingest"

# Explicit lmdeploy backends — never use *-auto-engine (would pick vllm on Linux).
LMDEPLOY_BACKENDS = frozenset(
    {
        "hybrid-lmdeploy-engine",
        "vlm-lmdeploy-engine",
    }
)
DEFAULT_BACKEND = "hybrid-lmdeploy-engine"

SKIP_BLOCK_TYPES = frozenset(
    {
        "header",
        "footer",
        "page_number",
        "page_footnote",
        "aside_text",
        "discarded",
    }
)

_log = logging.getLogger(__name__)
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_TAG_RE = re.compile(r"<[^>]+>")


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("tr", "td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._row.append(" ".join(self._cell).strip())
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = []

    def handle_data(self, data: str) -> None:
        self._cell.append(data.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MinerU PDF ingest (lmdeploy VLM) for breast-cancer RAG."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Single PDF to ingest (default: {DEFAULT_PDF.name}).",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Ingest every PDF in this folder (overrides --pdf).",
    )
    parser.add_argument(
        "--model-source",
        choices=("huggingface", "modelscope", "local"),
        default=None,
        help="Model hub (default: huggingface). Use modelscope if HF is slow.",
    )
    parser.add_argument(
        "--download-models-only",
        action="store_true",
        help="Download MinerU VLM (+ pipeline for hybrid) weights and exit.",
    )
    parser.add_argument(
        "--skip-model-prefetch",
        action="store_true",
        help="Skip upfront model download check (models must already be cached).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root (subdir per PDF stem).",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(LMDEPLOY_BACKENDS),
        default=DEFAULT_BACKEND,
        help=(
            "MinerU backend with lmdeploy inference. "
            "hybrid-lmdeploy-engine: best for tables + formulas (recommended). "
            "vlm-lmdeploy-engine: VLM-only."
        ),
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Document language for hybrid/pipeline OCR (en for medical papers).",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "txt", "ocr"),
        default="auto",
        help="PDF parse method (hybrid backend only).",
    )
    parser.add_argument(
        "--cache-max-entry-count",
        type=float,
        default=0.5,
        help="lmdeploy GPU memory fraction (0–1). Lower if OOM.",
    )
    parser.add_argument(
        "--lmdeploy-backend",
        choices=("auto", "pytorch", "turbomind"),
        default="auto",
        help=(
            "lmdeploy engine. auto uses turbomind on WSL when gcc is missing "
            "(avoids Triton 'Failed to find C compiler'). pytorch needs build-essential."
        ),
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="First page index (0-based).",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Last page index (0-based, inclusive).",
    )
    parser.add_argument(
        "--no-formulas",
        action="store_true",
        help="Disable formula parsing.",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table parsing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1800,
        help="Approx max characters per text chunk.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs that already have document.md.",
    )
    return parser.parse_args()


def _require_lmdeploy() -> None:
    try:
        import lmdeploy  # noqa: F401
    except ImportError as exc:
        _log.error(
            "lmdeploy is not installed. Install with: uv pip install \"mineru[core,lmdeploy]\""
        )
        raise SystemExit(2) from exc


def _setup_download_env() -> None:
    """Speed up / clarify Hugging Face downloads when possible."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        _log.info("hf_transfer enabled for faster model downloads.")
    except ImportError:
        _log.info(
            "Tip: pip install hf_transfer for faster downloads, or use --model-source modelscope."
        )


def _apply_model_source(model_source: str | None) -> None:
    if model_source:
        os.environ["MINERU_MODEL_SOURCE"] = model_source
    source = os.getenv("MINERU_MODEL_SOURCE", "huggingface")
    _log.info("MinerU model source: %s", source)


def _configure_lmdeploy_backend(choice: str) -> None:
    """
    MinerU defaults to lmdeploy pytorch on CUDA sm>=8.0, which needs Triton + gcc.
    WSL often has no C compiler → use turbomind or install build-essential.
    """
    if os.getenv("MINERU_LMDEPLOY_BACKEND"):
        _log.info("lmdeploy backend (env): %s", os.environ["MINERU_LMDEPLOY_BACKEND"])
        return

    if choice in ("pytorch", "turbomind"):
        os.environ["MINERU_LMDEPLOY_BACKEND"] = choice
        _log.info("lmdeploy backend: %s", choice)
        return

    if shutil.which("gcc") or shutil.which("cc"):
        _log.info("lmdeploy backend: auto (MinerU may select pytorch on sm>=8.0)")
        return

    os.environ["MINERU_LMDEPLOY_BACKEND"] = "turbomind"
    _log.warning(
        "No C compiler (gcc) found — using lmdeploy turbomind to avoid Triton build errors. "
        "For pytorch backend: sudo apt install -y build-essential"
    )


def _ensure_mineru_models(backend: str) -> None:
    """
    Download models before parsing so progress is visible and parsing does not
    look stuck inside do_parse().
    """
    from mineru.cli.models_download import (
        download_pipeline_models,
        download_vlm_models,
    )
    from mineru.utils.enum_class import ModelPath

    need_pipeline = backend.startswith("hybrid-")
    _log.info(
        "Checking MinerU model cache (VLM ~2.3GB%s). "
        "First download can take 10–60+ minutes depending on bandwidth. "
        "Ctrl+C is safe — re-run to resume.",
        " + pipeline models" if need_pipeline else "",
    )
    _log.info("VLM repo: %s", ModelPath.vlm_root_hf)

    download_vlm_models()
    if need_pipeline:
        _log.info("Downloading pipeline models (required for hybrid backend)...")
        download_pipeline_models()

    _log.info("MinerU models are ready.")


def _collect_pdfs(pdf: Path, pdf_dir: Path | None) -> list[Path]:
    if pdf_dir is not None:
        root = pdf_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"PDF directory not found: {root}")
        paths = sorted(root.glob("*.pdf"))
        if not paths:
            raise FileNotFoundError(f"No PDFs in {root}")
        return paths
    path = pdf.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    return [path]


def _mineru_parse_dir(out_root: Path, stem: str, method: str) -> Path:
    return out_root / stem / f"hybrid_{method}"


def _run_mineru(
    pdf_path: Path,
    work_root: Path,
    *,
    backend: str,
    lang: str,
    method: str,
    formula_enable: bool,
    table_enable: bool,
    start_page: int,
    end_page: int | None,
    cache_max_entry_count: float,
) -> Path:
    from mineru.cli.common import do_parse, read_fn

    stem = pdf_path.stem
    os.environ.setdefault("MINERU_LMDEPLOY_DEVICE", "cuda")

    _log.info(
        "MinerU parse: %s backend=%s lang=%s (lmdeploy)",
        pdf_path.name,
        backend,
        lang,
    )
    do_parse(
        output_dir=str(work_root),
        pdf_file_names=[stem],
        pdf_bytes_list=[read_fn(pdf_path)],
        p_lang_list=[lang],
        backend=backend,
        parse_method=method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        start_page_id=start_page,
        end_page_id=end_page,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_content_list=True,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        cache_max_entry_count=cache_max_entry_count,
    )
    parse_dir = _mineru_parse_dir(work_root, stem, method)
    md_path = parse_dir / f"{stem}.md"
    if not md_path.is_file():
        raise FileNotFoundError(f"MinerU did not produce markdown at {md_path}")
    return parse_dir


def _html_table_to_markdown(html: str) -> str:
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return _TAG_RE.sub(" ", html).strip()
    if not parser.rows:
        return _TAG_RE.sub(" ", html).strip()
    lines: list[str] = []
    ncol = 0
    for row in parser.rows:
        ncol = max(ncol, len(row))
        lines.append("| " + " | ".join(cell or " " for cell in row) + " |")
    if len(lines) >= 2 and ncol > 0:
        sep = "| " + " | ".join(["---"] * ncol) + " |"
        lines.insert(1, sep)
    return "\n".join(lines)


def _block_text(block: dict[str, Any]) -> str:
    btype = block.get("type", "")
    if btype == "text":
        return str(block.get("text", "")).strip()
    if btype == "list":
        items = block.get("list_items") or []
        return "\n".join(str(i).strip() for i in items if i)
    if btype == "equation":
        return str(block.get("text", "")).strip()
    if btype == "code":
        body = block.get("code_body") or ""
        caps = block.get("code_caption") or []
        cap = "\n".join(caps).strip()
        return f"{cap}\n{body}".strip() if cap else str(body).strip()
    if btype == "table":
        caps = block.get("table_caption") or []
        cap = " ".join(str(c) for c in caps).strip()
        body = block.get("table_body") or ""
        md = _html_table_to_markdown(str(body)) if body else ""
        return f"{cap}\n\n{md}".strip() if cap else md
    if btype in ("image", "chart"):
        caps = block.get("image_caption") or block.get("chart_caption") or []
        return " ".join(str(c) for c in caps).strip()
    return str(block.get("text", "")).strip()


def _load_content_list(parse_dir: Path, stem: str) -> list[dict[str, Any]]:
    for name in (f"{stem}_content_list_v2.json", f"{stem}_content_list.json"):
        path = parse_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return []


def _extract_tables(content_list: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for block in content_list:
        if block.get("type") != "table":
            continue
        caps = block.get("table_caption") or []
        title = " ".join(str(c) for c in caps).strip() or None
        html = str(block.get("table_body") or "")
        md = _html_table_to_markdown(html)
        tables.append(
            {
                "index": len(tables),
                "title": title,
                "page": block.get("page_idx"),
                "source": source,
                "markdown": md,
                "html": html or None,
            }
        )
    return tables


def _split_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if size + len(para) + 2 > max_chars and buf:
            parts.append("\n\n".join(buf))
            buf = []
            size = 0
        if len(para) > max_chars:
            if buf:
                parts.append("\n\n".join(buf))
                buf = []
                size = 0
            for i in range(0, len(para), max_chars):
                parts.append(para[i : i + max_chars])
            continue
        buf.append(para)
        size += len(para) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def _heading_level(block: dict[str, Any]) -> int | None:
    level = block.get("text_level")
    if level is None:
        return None
    try:
        lvl = int(level)
    except (TypeError, ValueError):
        return None
    return lvl if lvl > 0 else None


def build_chunks(
    content_list: list[dict[str, Any]],
    *,
    source: str,
    max_chars: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    section = ""
    text_buf: list[str] = []

    def flush_text() -> None:
        nonlocal text_buf
        joined = "\n\n".join(text_buf).strip()
        text_buf = []
        if not joined:
            return
        for part in _split_text(joined, max_chars):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "type": "text",
                    "text": part,
                    "source": source,
                    "section": section or None,
                    "page": page_hint,
                }
            )

    page_hint: int | None = None
    for block in content_list:
        btype = block.get("type", "")
        if btype in SKIP_BLOCK_TYPES:
            continue
        page_hint = block.get("page_idx", page_hint)
        if btype == "text" and _heading_level(block) is not None:
            flush_text()
            section = str(block.get("text", "")).strip()
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "type": "heading",
                    "text": section,
                    "source": source,
                    "section": section,
                    "page": page_hint,
                    "level": _heading_level(block),
                }
            )
            continue
        if btype == "table":
            flush_text()
            text = _block_text(block)
            if text:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "type": "table",
                        "text": text,
                        "source": source,
                        "section": section or None,
                        "page": page_hint,
                    }
                )
            continue
        if btype == "equation":
            flush_text()
            text = _block_text(block)
            if text:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "type": "equation",
                        "text": text,
                        "source": source,
                        "section": section or None,
                        "page": page_hint,
                    }
                )
            continue
        if btype in ("image", "chart"):
            caption = _block_text(block)
            if caption:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "type": btype,
                        "text": caption,
                        "source": source,
                        "section": section or None,
                        "page": page_hint,
                    }
                )
            continue
        piece = _block_text(block)
        if piece:
            text_buf.append(piece)
    flush_text()
    return chunks


def _publish_outputs(
    parse_dir: Path,
    stem: str,
    out_dir: Path,
    *,
    max_chars: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = f"{stem}.pdf"

    md_src = parse_dir / f"{stem}.md"
    document_md = normalize_markdown(md_src.read_text(encoding="utf-8"))
    (out_dir / "document.md").write_text(document_md, encoding="utf-8")

    content_list = _load_content_list(parse_dir, stem)
    if content_list:
        (out_dir / "content_list.json").write_text(
            json.dumps(content_list, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    tables = _extract_tables(content_list, source)
    write_tables_json(tables, out_dir / "tables.json")

    chunks = build_chunks(content_list, source=source, max_chars=max_chars)
    (out_dir / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    raw_link = out_dir / "mineru"
    if raw_link.exists() or raw_link.is_symlink():
        if raw_link.is_symlink():
            raw_link.unlink()
        elif raw_link.is_dir():
            shutil.rmtree(raw_link)
    try:
        raw_link.symlink_to(parse_dir.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(parse_dir, raw_link, dirs_exist_ok=True)

    _log.info(
        "Wrote %s (md=%d chars, tables=%d, chunks=%d)",
        out_dir,
        len(document_md),
        len(tables),
        len(chunks),
    )


def ingest_pdf(pdf_path: Path, args: argparse.Namespace) -> int:
    stem = pdf_path.stem
    out_dir = (args.out_dir / stem).resolve()
    if args.skip_existing and (out_dir / "document.md").is_file():
        _log.info("Skipping %s (document.md exists)", stem)
        return 0

    work_root = out_dir / "_mineru_work"
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        parse_dir = _run_mineru(
            pdf_path,
            work_root,
            backend=args.backend,
            lang=args.lang,
            method=args.method,
            formula_enable=not args.no_formulas,
            table_enable=not args.no_tables,
            start_page=args.start_page,
            end_page=args.end_page,
            cache_max_entry_count=args.cache_max_entry_count,
        )
        _publish_outputs(parse_dir, stem, out_dir, max_chars=args.chunk_size)
    finally:
        if work_root.is_dir():
            shutil.rmtree(work_root, ignore_errors=True)

    root_md = PROJECT_ROOT / "output.md"
    root_md.write_text((out_dir / "document.md").read_text(encoding="utf-8"), encoding="utf-8")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _require_lmdeploy()
    _setup_download_env()
    args = parse_args()
    _apply_model_source(args.model_source)
    _configure_lmdeploy_backend(args.lmdeploy_backend)

    if args.backend not in LMDEPLOY_BACKENDS:
        _log.error("Backend must be one of %s (got %s)", LMDEPLOY_BACKENDS, args.backend)
        return 1

    if not args.skip_model_prefetch:
        try:
            _ensure_mineru_models(args.backend)
        except KeyboardInterrupt:
            _log.warning(
                "Model download interrupted. Re-run the same command to resume "
                "(Hugging Face keeps partial files)."
            )
            return 130
        except Exception as exc:
            _log.error("Model download failed: %s", exc)
            return 1

    if args.download_models_only:
        _log.info("Models downloaded. Run again without --download-models-only to ingest.")
        return 0

    try:
        pdfs = _collect_pdfs(args.pdf, args.pdf_dir)
    except FileNotFoundError as exc:
        _log.error("%s", exc)
        return 1

    failures = 0
    for pdf_path in pdfs:
        try:
            ingest_pdf(pdf_path, args)
        except Exception as exc:
            failures += 1
            _log.exception("Failed %s: %s", pdf_path.name, exc)

    if failures:
        _log.error("%d of %d PDF(s) failed", failures, len(pdfs))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Extract PDF tables with Camelot for side-by-side ingest testing.

This script writes:
- output/ingest/<pdf_stem>/camelot-<flavor>-tables.json   (filtered + cleaned)
- output/ingest/<pdf_stem>/camelot-<flavor>-tables.md
- output/ingest/<pdf_stem>/camelot-<flavor>-raw.json      (all Camelot hits, with --keep-raw)

Notes:
- Camelot works on text-based PDFs (not scanned-image PDFs).
- `stream` fits journal tables without grid lines; `lattice` needs visible borders.
- Keeps all paper tables (TABLE 1, TABLE 2, …); drops 1-column prose false positives.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_INGEST_DIR = Path(__file__).resolve().parent
if str(_INGEST_DIR) not in sys.path:
    sys.path.insert(0, str(_INGEST_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "ingest"
DEFAULT_PDF = PROJECT_ROOT / "papers" / "test_ingest" / "19-STS729.pdf"

_log = logging.getLogger(__name__)
_TABLE_CAPTION_RE = re.compile(r"^TABLE\s+\d+\b", re.I)
_HEADER_PATTERNS = (
    re.compile(r"category.*hazard\s*ratio|hazard\s*ratio.*category", re.I),
    re.compile(r"term.*o/e|o/e.*term", re.I),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camelot table extraction test for PDF ingest."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Input PDF path.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (subdir per PDF stem).",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default="all",
        help="Pages to parse, e.g. '1', '1,3', '1-end', 'all'.",
    )
    parser.add_argument(
        "--flavor",
        choices=("lattice", "stream", "auto"),
        default="auto",
        help="Camelot parser (auto: lattice then stream if zero tables).",
    )
    parser.add_argument("--parallel", action="store_true", help="Parse pages in parallel.")
    parser.add_argument(
        "--min-cols",
        type=int,
        default=2,
        help="Minimum columns for a detection (use 2 for calibration-style tables).",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=3,
        help="Drop detections with fewer data rows.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Also write unfiltered Camelot output to camelot-<flavor>-raw.json.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable prose filtering (not recommended for papers).",
    )
    parser.add_argument(
        "--line-scale",
        type=int,
        default=40,
        help="Lattice line_scale tuning (larger helps detect small lines).",
    )
    parser.add_argument(
        "--copy-text",
        type=str,
        default="h",
        help="Lattice copy_text option (h, v, or h,v).",
    )
    return parser.parse_args()


def write_tables_markdown(records: list[dict[str, Any]], path: Path) -> None:
    parts: list[str] = []
    for rec in records:
        page = rec.get("page")
        page_note = f", page {page}" if page is not None else ""
        report = rec.get("parsing_report") or {}
        acc = report.get("accuracy")
        acc_note = f", accuracy {acc:.2f}" if isinstance(acc, (int, float)) else ""
        title = rec.get("title")
        title_note = f" — {title}" if title else ""
        parts.append(f"## Table {rec['index']}{page_note}{acc_note}{title_note}\n")
        parts.append(rec.get("markdown", "").strip())
        parts.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _cell_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _row_text(df: pd.DataFrame, i: int) -> str:
    return " | ".join(_cell_str(df.iloc[i, j]) for j in range(df.shape[1]))


def _is_prose_table(df: pd.DataFrame) -> bool:
    """Camelot stream often treats paragraph blocks as single-column tables."""
    if df.shape[1] == 1:
        lengths = df.iloc[:, 0].map(lambda v: len(_cell_str(v)))
        return bool(lengths.median() > 60 or lengths.max() > 200)
    if df.shape[1] == 2:
        long_rows = 0
        for i in range(len(df)):
            a, b = _cell_str(df.iloc[i, 0]), _cell_str(df.iloc[i, 1])
            if len(a) > 90 and len(b) > 90 and not re.search(r"\d", a[:30]):
                long_rows += 1
        return long_rows >= max(3, len(df) // 2)
    return False


def _has_table_caption(df: pd.DataFrame, scan_rows: int = 6) -> bool:
    for i in range(min(scan_rows, len(df))):
        for j in range(df.shape[1]):
            if _TABLE_CAPTION_RE.match(_cell_str(df.iloc[i, j])):
                return True
    return False


def _find_header_row(df: pd.DataFrame) -> int | None:
    for i in range(len(df)):
        row_text = _row_text(df, i)
        if not any(pat.search(row_text) for pat in _HEADER_PATTERNS):
            continue
        cells = [_cell_str(df.iloc[i, j]) for j in range(df.shape[1])]
        # Reject prose rows that happen to mention "term" and "o/e".
        if any(len(c) > 80 for c in cells):
            continue
        return i
    return None


def truncate_table_tail(df: pd.DataFrame) -> pd.DataFrame:
    """Drop body-text rows Camelot often attaches after the real table grid."""
    if len(df) < 4:
        return df
    for i in range(3, len(df)):
        cells = [_cell_str(df.iloc[i, j]) for j in range(df.shape[1])]
        if all(len(c) > 100 for c in cells if c) and not any(
            re.search(r"^\d[\d,]*$", c.split("\n")[0]) for c in cells if c
        ):
            return df.iloc[:i].copy()
    return df


def promote_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """Use the row containing Category + Hazard ratio as column headers."""
    work = df.copy()
    work.columns = list(range(work.shape[1]))
    header_idx = _find_header_row(work)
    if header_idx is None:
        return df

    headers: list[str] = []
    for j in range(work.shape[1]):
        name = _cell_str(work.iloc[header_idx, j])
        if not name and j == 0:
            name = "Factor"
        elif not name:
            name = f"col_{j}"
        headers.append(name)

    body = work.iloc[header_idx + 1 :].reset_index(drop=True)
    body.columns = headers
    return body


def _table_title(df: pd.DataFrame) -> str | None:
    blob = " ".join(_cell_str(v) for v in df.astype(str).values.flatten()[:12])
    if m := re.search(r"TABLE\s+\d+", blob, re.I):
        return m.group(0)
    if "Summary of risk factor" in blob:
        return "TABLE 1 — Summary of risk factor parameters"
    return None


def _safe_fill_labels(df: pd.DataFrame, fill_labels: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    if df.empty or df.shape[1] == 0:
        return df
    first = str(df.columns[0])
    if first not in df.columns:
        return df
    try:
        return fill_labels(df)
    except KeyError:
        return df


def postprocess_camelot_df(
    df: pd.DataFrame,
    *,
    normalize_df: Callable[[pd.DataFrame], pd.DataFrame],
    fill_labels: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    df = truncate_table_tail(normalize_df(df))
    df = promote_header_row(df)
    return _safe_fill_labels(df, fill_labels)


def is_likely_real_table(
    df: pd.DataFrame,
    *,
    min_cols: int,
    min_rows: int,
) -> bool:
    if _is_prose_table(df):
        return False
    if df.shape[1] < min_cols or len(df) < min_rows:
        return False
    if _has_table_caption(df) or _find_header_row(df) is not None:
        return True
    numeric_cells = 0
    for v in df.astype(str).values.flatten():
        s = _cell_str(v).split("\n")[0]
        if re.fullmatch(r"[\d.,]+", s) or re.match(r"^[\d.]+", s):
            numeric_cells += 1
    return numeric_cells >= max(6, len(df))


def dedupe_table_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the richest detection when Camelot splits the same TABLE N twice."""
    by_title: dict[str, dict[str, Any]] = {}
    untitled: list[dict[str, Any]] = []
    for rec in records:
        title = rec.get("title")
        if not title:
            untitled.append(rec)
            continue
        key = re.sub(r"\s+", " ", title.upper())
        prev = by_title.get(key)
        if prev is None or rec["rows"] > prev["rows"]:
            by_title[key] = rec
    merged = list(by_title.values()) + untitled
    merged.sort(key=lambda r: (r.get("page") or 0, r["index"]))
    for i, rec in enumerate(merged):
        rec["index"] = i
    return merged


def _table_record(
    index: int,
    table: Any,
    *,
    normalize_df: Callable[[pd.DataFrame], pd.DataFrame],
    fill_labels: Callable[[pd.DataFrame], pd.DataFrame],
    postprocess: bool,
) -> dict[str, Any]:
    raw_df = normalize_df(table.df)
    df = (
        postprocess_camelot_df(raw_df, normalize_df=lambda x: x, fill_labels=fill_labels)
        if postprocess
        else raw_df
    )
    report = getattr(table, "parsing_report", {}) or {}
    return {
        "index": index,
        "rows": len(df),
        "columns": [str(c) for c in df.columns.tolist()],
        "data": json.loads(df.to_json(orient="records")),
        "markdown": df.to_markdown(index=False),
        "parsing_report": report,
        "page": report.get("page"),
        "title": _table_title(raw_df),
    }


def _read_camelot(camelot: Any, source_path: Path, flavor: str, args: argparse.Namespace) -> Any:
    read_kwargs: dict[str, Any] = {
        "pages": args.pages,
        "flavor": flavor,
        "parallel": args.parallel,
    }
    if flavor == "lattice":
        read_kwargs["line_scale"] = args.line_scale
        read_kwargs["copy_text"] = [
            part.strip() for part in args.copy_text.split(",") if part.strip()
        ]
    return camelot.read_pdf(str(source_path), **read_kwargs)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    source_path = args.pdf.resolve()
    if not source_path.is_file():
        _log.error("PDF not found: %s", source_path)
        return 1

    try:
        import camelot  # type: ignore
    except Exception as exc:
        _log.error("Camelot is not installed or failed to import: %s", exc)
        _log.error("Install with: uv add camelot-py[base]")
        return 2
    try:
        from ingest_utils import fill_merged_row_labels, normalize_dataframe, write_tables_json
    except Exception as exc:
        _log.error("Missing ingest dependencies: %s", exc)
        return 2

    flavors: list[str]
    if args.flavor == "auto":
        flavors = ["lattice", "stream"]
    else:
        flavors = [args.flavor]

    tables = None
    used_flavor = flavors[-1]
    for flavor in flavors:
        _log.info("Camelot extract: pdf=%s flavor=%s pages=%s", source_path.name, flavor, args.pages)
        try:
            tables = _read_camelot(camelot, source_path, flavor, args)
        except Exception as exc:
            _log.error("Camelot extraction failed (%s): %s", flavor, exc)
            return 3
        used_flavor = flavor
        if len(tables) > 0 or args.flavor != "auto":
            break
        _log.warning("Camelot %s found 0 tables; trying next flavor.", flavor)

    if tables is None or len(tables) == 0:
        _log.warning("No tables found with flavors: %s", flavors)
        return 0

    out_dir = (args.out_dir / source_path.stem).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"camelot-{used_flavor}-tables"
    out_json = out_dir / f"{stem}.json"
    out_md = out_dir / f"{stem}.md"
    out_raw = out_dir / f"camelot-{used_flavor}-raw.json"

    all_records = [
        _table_record(
            i,
            table,
            normalize_df=normalize_dataframe,
            fill_labels=fill_merged_row_labels,
            postprocess=False,
        )
        for i, table in enumerate(tables)
    ]

    if args.no_filter:
        records = [
            _table_record(
                i,
                table,
                normalize_df=normalize_dataframe,
                fill_labels=fill_merged_row_labels,
                postprocess=True,
            )
            for i, table in enumerate(tables)
        ]
        rejected = 0
    else:
        candidates: list[dict[str, Any]] = []
        rejected = 0
        for i, table in enumerate(tables):
            raw_df = normalize_dataframe(table.df)
            if not is_likely_real_table(
                raw_df, min_cols=args.min_cols, min_rows=args.min_rows
            ):
                rejected += 1
                continue
            candidates.append(
                _table_record(
                    i,
                    table,
                    normalize_df=normalize_dataframe,
                    fill_labels=fill_merged_row_labels,
                    postprocess=True,
                )
            )
        records = dedupe_table_records(candidates)

    write_tables_json(records, out_json)
    write_tables_markdown(records, out_md)
    if args.keep_raw:
        write_tables_json(all_records, out_raw)

    _log.info("Wrote %s (%d paper tables)", out_json, len(records))
    _log.info("Wrote %s", out_md)
    if rejected:
        _log.info(
            "Filtered out %d prose/false-positive detections (of %d raw). "
            "Use --keep-raw to inspect everything.",
            rejected,
            len(all_records),
        )
    if args.keep_raw:
        _log.info("Wrote %s (%d raw)", out_raw, len(all_records))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

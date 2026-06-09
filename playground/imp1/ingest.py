#!/usr/bin/env python3
"""
ingest.py — CLI entry point for the breast-cancer RAG ingestion pipeline.

Usage
─────
  # Parse all PDFs in data/pdfs/
  python ingest.py

  # Re-parse even if output already exists
  python ingest.py --overwrite

  # Custom PDF directory
  python ingest.py --pdf-dir /path/to/papers

  # Quick sanity-check: print first 3 chunks of a parsed paper
  python ingest.py --preview output/parsed/smith2021/chunks.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LOG_LEVEL, OUTPUT_DIR, PDF_DIR
from parse_pdfs import run_ingestion


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep docling at INFO so pipeline progress is visible (formula batches, etc.).
    # medical-a1 does the same; suppressing docling makes long converts look stuck.
    for noisy in ("huggingface_hub", "transformers", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _preview(jsonl_path: Path, n: int = 3) -> None:
    """Print first *n* chunks for quick inspection."""
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            print(
                f"\n── Chunk {obj['chunk_index']} [{obj['chunk_type'].upper()}] "
                f"│ section={obj['section']} │ page={obj['page_start']} ──"
            )
            print(f"   Title  : {obj.get('title', 'n/a')}")
            print(f"   DOI    : {obj.get('doi', 'n/a')}")
            print(f"   Authors: {', '.join(obj.get('authors', []))}")
            if obj["chunk_type"] == "table":
                print(f"   Table  :\n{obj['table_markdown'][:300]}")
            elif obj["chunk_type"] == "formula":
                print(f"   LaTeX  : {obj['formula_latex']}")
            elif obj["chunk_type"] == "figure":
                print(f"   Caption: {obj['figure_caption']}")
                print(f"   Artifact: {obj['artifact_path']}")
            else:
                print(f"   Text   : {obj['text'][:300].replace(chr(10), ' ')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Breast-cancer RAG — ingestion pipeline"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=PDF_DIR,
        help=f"PDF input folder (default: {PDF_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output root (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-parse already-processed PDFs"
    )
    parser.add_argument(
        "--preview",
        type=Path,
        metavar="JSONL",
        help="Print first 3 chunks of a JSONL and exit",
    )
    parser.add_argument(
        "--log-level", default=LOG_LEVEL, choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)

    if args.preview:
        _preview(args.preview)
        return

    run_ingestion(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

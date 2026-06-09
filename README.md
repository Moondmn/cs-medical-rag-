# rag-uni-a1

Breast-cancer RAG project — PDF ingestion with [Docling](https://github.com/docling-project/docling).

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended)
- **CUDA GPU** (optional but strongly recommended — CPU runs are much slower)
- **~5 GB disk** for Docling model weights (downloaded on first run)

## Setup

```bash
# Clone and enter the repo
cd rag-uni-a1

# Install dependencies (PyTorch CUDA wheels on Linux/Windows)
uv sync

# Optional: prefetch Docling models while online
uv run docling-tools models download

# Optional: faster Hugging Face downloads
export HF_TOKEN=hf_...
```

OCR fallback chain (used by `imp1`): EasyOCR → RapidOCR → Tesseract CLI. RapidOCR is included via Docling; install EasyOCR or Tesseract only if you want a different engine.

---

## `playground/imp1` — RAG ingestion pipeline

Full parse → **RAG-ready `chunks.jsonl`** with metadata (title, authors, DOI, IMRaD sections), plus markdown, tables, formulas, figures, and PNG artifacts.

**Defaults** (see `playground/imp1/config.py`):

| Setting | Value |
|---------|-------|
| Input PDFs | `data/pdfs/` |
| Output | `output/parsed/<paper-stem>/` |

### Run

```bash
# Parse all PDFs in the default folder
uv run python playground/imp1/ingest.py

# Custom input folder
uv run python playground/imp1/ingest.py --pdf-dir ./papers/test_ingest/

# Re-parse papers that were already processed
uv run python playground/imp1/ingest.py --pdf-dir ./papers/test_ingest/ --overwrite

# Preview first 3 chunks from a parsed paper
uv run python playground/imp1/ingest.py --preview output/parsed/BC_Validation_2019/chunks.jsonl
```

### Outputs per paper

```
output/parsed/<stem>/
  chunks.jsonl      # one JSON object per text / table / formula / figure chunk
  document.md
  tables.json
  formulas.json
  pictures.json
  document.json
  artifacts/
    pages/          # full-page PNGs
    pictures/       # figure crops
    tables/         # table crops
```

### Notes

- First run downloads layout, table, and formula models — allow several minutes.
- **Formula enrichment** (`CodeFormulaV2`) is slow (~30 s per batch of images). Docling INFO logs show progress; a single paper can take **5–10+ minutes**.
- Tune speed/quality in `playground/imp1/config.py` (`PIPELINE_USE_OCR`, `PIPELINE_IMAGES_SCALE`, etc.).

---

## `playground/ingest/docling-a1.py` — parse + confidence + picture captions

Lighter experiment script: Docling parse with **SmolVLM figure captions**, **confidence grades**, and markdown export. No chunking.

**Hard-coded paths** in the script:

| Setting | Value |
|---------|-------|
| Input PDFs | `papers/test_ingest/` |
| Output | `output/ingest/<paper-stem>/` |

### Run

```bash
uv run python playground/ingest/docling-a1.py
```

### Outputs per paper

```
output/ingest/<stem>/
  document.md
  document.json
  pictures.json
  confidence.json          # full Docling confidence report
  confidence_summary.json    # mean/low grades per page
  artifacts/pictures/
```

Also writes `output.md` at the repo root (copy of the last `document.md`).

### Notes

- Pipeline uses **picture description** (SmolVLM), **TableFormer ACCURATE** mode, and formula enrichment — expect long runtimes on CPU.
- The script currently processes **only the first PDF** in `papers/test_ingest/` and converts **pages 5–7** (`page_range=(5, 7)` in `ingest_pdf`). Edit those lines in `docling-a1.py` for a full batch or full document.

---

## Quick comparison

| | `imp1` | `docling-a1.py` |
|---|--------|-----------------|
| Purpose | RAG chunking + metadata | Parse quality / confidence probe |
| Main output | `chunks.jsonl` | `document.md` + `confidence.json` |
| Picture captions | No | Yes (SmolVLM) |
| Default PDF dir | `data/pdfs/` | `papers/test_ingest/` |
| CLI flags | `--pdf-dir`, `--overwrite`, `--preview` | None (edit script) |

"""
Central config for the breast-cancer RAG ingestion pipeline.
Edit here only — nothing else needs to change.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # repo root (imp1 → playground → repo)
PDF_DIR = ROOT / "data" / "pdfs"  # drop PDFs here
OUTPUT_DIR = ROOT / "output" / "parsed"  # JSONL + artifact dirs land here

# ── Pipeline toggles ──────────────────────────────────────────────────────────
# True  → try EasyOCR → RapidOCR → Tesseract → OcrAuto (for scanned / mixed PDFs)
# False → skip OCR entirely (pure digital text-layer PDFs only, faster)
PIPELINE_USE_OCR = True
PIPELINE_IMAGES_SCALE = 3.0  # DPI multiplier for PNG exports
PIPELINE_OCR_LANG = "eng"  # Tesseract language code fallback

# ── Section heading normalisation ─────────────────────────────────────────────
SECTION_LABEL_MAP: dict[str, str] = {
    "abstract": "abstract",
    "summary": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "background and introduction": "introduction",
    "methods": "methods",
    "materials and methods": "methods",
    "patients and methods": "methods",
    "study design": "methods",
    "methodology": "methods",
    "statistical analysis": "methods",
    "statistical methods": "methods",
    "results": "results",
    "findings": "results",
    "outcomes": "results",
    "discussion": "discussion",
    "interpretation": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "concluding remarks": "conclusion",
    "acknowledgements": "acknowledgements",
    "acknowledgments": "acknowledgements",
    "references": "references",
    "supplementary": "supplementary",
    "appendix": "supplementary",
    "supplementary material": "supplementary",
    "conflict of interest": "disclosures",
    "conflicts of interest": "disclosures",
    "disclosures": "disclosures",
    "funding": "disclosures",
    "author contributions": "author_contributions",
    "data availability": "data_availability",
    "data availability statement": "data_availability",
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

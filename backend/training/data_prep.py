"""
training/data_prep.py — Download and prepare the ec-darkpattern dataset
========================================================================
Downloads the yamanalab/ec-darkpattern dataset from GitHub and prepares
a CSV with 'text' and 'label' columns suitable for fine-tuning.

Usage:
    python training/data_prep.py
"""

from __future__ import annotations
import io
import os
import sys
import csv
import json
import zipfile
import requests
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_REPO = "https://github.com/yamanalab/ec-darkpattern/archive/refs/heads/main.zip"

# Label mapping (Mathur et al. category → numeric + our category)
LABEL_MAP = {
    # dark-pattern categories
    "urgency":                  ("urgency",         1),
    "scarcity":                 ("urgency",         1),
    "misdirection":             ("confirm_shaming", 1),
    "confirm_shaming":          ("confirm_shaming", 1),
    "forced_continuity":        ("hidden_flows",    1),
    "hidden_costs":             ("hidden_flows",    1),
    "disguised_ads":            ("disguised_ads",   1),
    "trick_questions":          ("optin",           1),
    "sneak_into_basket":        ("optin",           1),
    # benign
    "not_dark_pattern":         ("none",            0),
    "normal":                   ("none",            0),
    "":                         ("none",            0),
}


def download_dataset() -> Path:
    """Download the ec-darkpattern repo zip."""
    zip_path = DATA_DIR / "ec-darkpattern.zip"
    if zip_path.exists():
        print(f"[data_prep] Found cached zip at {zip_path}")
        return zip_path

    print(f"[data_prep] Downloading ec-darkpattern dataset...")
    response = requests.get(DATASET_REPO, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"[data_prep] Downloaded to {zip_path}")
    return zip_path


def extract_dataset(zip_path: Path) -> Path:
    """Extract and locate the main CSV/JSON data files."""
    extract_dir = DATA_DIR / "ec-darkpattern-raw"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        # Rename extracted folder
        extracted = list(DATA_DIR.glob("ec-darkpattern-*"))
        if extracted:
            extracted[0].rename(extract_dir)

    return extract_dir


def build_dataframe(raw_dir: Path) -> pd.DataFrame:
    """
    Parse all CSV files in the dataset into a unified DataFrame
    with columns: text, category, label (0=clean, 1=dark pattern).
    """
    records = []

    # Look for CSV files anywhere in the raw directory
    csv_files = list(raw_dir.rglob("*.csv"))
    json_files = list(raw_dir.rglob("*.json"))

    print(f"[data_prep] Found {len(csv_files)} CSV files, {len(json_files)} JSON files")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip")
            # Try common column names
            text_col  = next((c for c in df.columns if "text" in c.lower()), None)
            label_col = next((c for c in df.columns if "label" in c.lower() or "category" in c.lower() or "type" in c.lower()), None)

            if text_col is None:
                continue

            for _, row in df.iterrows():
                text = str(row.get(text_col, "")).strip()
                if not text or text == "nan":
                    continue

                raw_label = str(row.get(label_col, "not_dark_pattern")).lower().strip() if label_col else "not_dark_pattern"
                raw_label = raw_label.replace(" ", "_")

                # Map to our categories
                category, numeric = LABEL_MAP.get(raw_label, ("none", 0))

                # Try to infer from column name if label col is missing
                if not label_col:
                    fn = csv_file.stem.lower()
                    if any(dp in fn for dp in ("urgency", "scarcity")):
                        category, numeric = "urgency", 1
                    elif any(dp in fn for dp in ("misdirect", "shaming")):
                        category, numeric = "confirm_shaming", 1
                    elif "ad" in fn:
                        category, numeric = "disguised_ads", 1
                    elif any(dp in fn for dp in ("forced", "hidden")):
                        category, numeric = "hidden_flows", 1

                records.append({
                    "text":     text,
                    "category": category,
                    "label":    numeric,
                    "source":   csv_file.stem,
                })
        except Exception as e:
            print(f"[data_prep] Warning: could not parse {csv_file.name}: {e}")

    if not records:
        # Create a small synthetic seed dataset if download failed
        print("[data_prep] No records found — generating synthetic seed data")
        records = _generate_synthetic_seed()

    df = pd.DataFrame(records)
    print(f"[data_prep] Total records: {len(df)} | Dark patterns: {df['label'].sum()}")
    return df


def _generate_synthetic_seed() -> list:
    """Minimal synthetic dataset based on known dark-pattern examples."""
    return [
        # urgency
        {"text": "Only 2 left in stock — order soon!", "category": "urgency", "label": 1, "source": "synthetic"},
        {"text": "Offer expires in 10:00 minutes!", "category": "urgency", "label": 1, "source": "synthetic"},
        {"text": "Flash sale ends tonight!", "category": "urgency", "label": 1, "source": "synthetic"},
        {"text": "HURRY! Only 3 items remaining.", "category": "urgency", "label": 1, "source": "synthetic"},
        {"text": "Last chance — deal expires at midnight!", "category": "urgency", "label": 1, "source": "synthetic"},
        # confirm_shaming
        {"text": "No thanks, I hate saving money.", "category": "confirm_shaming", "label": 1, "source": "synthetic"},
        {"text": "No, I don't want better skin.", "category": "confirm_shaming", "label": 1, "source": "synthetic"},
        {"text": "Decline this amazing free offer.", "category": "confirm_shaming", "label": 1, "source": "synthetic"},
        {"text": "I'm fine paying full price.", "category": "confirm_shaming", "label": 1, "source": "synthetic"},
        # optin
        {"text": "Yes, sign me up for exclusive deals!", "category": "optin", "label": 1, "source": "synthetic"},
        {"text": "I agree to receive marketing emails from partners.", "category": "optin", "label": 1, "source": "synthetic"},
        # hidden_flows
        {"text": "Cancel anytime (buried in fine print)", "category": "hidden_flows", "label": 1, "source": "synthetic"},
        {"text": "Unsubscribe", "category": "hidden_flows", "label": 1, "source": "synthetic"},
        # disguised_ads
        {"text": "Sponsored content by our partners", "category": "disguised_ads", "label": 1, "source": "synthetic"},
        {"text": "Recommended for you (Promoted)", "category": "disguised_ads", "label": 1, "source": "synthetic"},
        # clean examples
        {"text": "Add to cart", "category": "none", "label": 0, "source": "synthetic"},
        {"text": "View product details", "category": "none", "label": 0, "source": "synthetic"},
        {"text": "Free shipping on orders over $50", "category": "none", "label": 0, "source": "synthetic"},
        {"text": "Returns accepted within 30 days", "category": "none", "label": 0, "source": "synthetic"},
        {"text": "Customer reviews (verified purchases)", "category": "none", "label": 0, "source": "synthetic"},
    ]


def split_and_save(df: pd.DataFrame):
    """Split into train/val/test and save CSV files."""
    # Filter to binary: 0 vs 1
    df = df[["text", "label", "category"]].dropna()
    df = df[df["text"].str.len() >= 5]

    train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv",   index=False)
    test_df.to_csv(DATA_DIR / "test.csv",  index=False)

    print(f"[data_prep] Saved splits → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"[data_prep] Data directory: {DATA_DIR}")


def main():
    print("=" * 60)
    print("Dark Pattern Detector — Dataset Preparation")
    print("=" * 60)

    try:
        zip_path = download_dataset()
        raw_dir  = extract_dataset(zip_path)
        df       = build_dataframe(raw_dir)
    except Exception as e:
        print(f"[data_prep] Download/extract failed ({e}) — using synthetic seed only")
        records = _generate_synthetic_seed()
        df = pd.DataFrame(records)

    split_and_save(df)
    print("\n✅ Dataset preparation complete!")
    print(f"   Train/Val/Test CSVs saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()

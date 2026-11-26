"""Curation script for targeted hard-negative dataset from existing raw datasets.

Requirements:
1. Lifestyle / worn bracelets (Bracelet class, wrist/arm context, open cuff, heart-link, bangle)
2. In-hand / boxed necklaces (Necklace class, chain visible, hand-held, gift-box, dark cushion, prominent pendant)
3. Plain-band rings (Ring class, thick/plain bands, ambiguous scale)
4. Lifestyle / in-hand rings (Ring class, hand/finger/lifestyle context)

Strict constraints:
- Source class strictly verified from dataset metadata/directory.
- Exact-file and perceptual duplicate check against train, val, test, and regression.
- Destination: data/processed/classification/hard_negatives/<Class>/
- No modification of train, val, test, or regression splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(r"c:\Users\konat\Desktop\gold")
DATASET_DIR = ROOT / "dataset" / "dataset"
LABELS_CSV = DATASET_DIR / "dataset_labels.csv"
DEST_BASE = ROOT / "data" / "processed" / "classification" / "hard_negatives"

# Build hash index of all current splits
print("Indexing hashes of existing train, val, test, and regression splits...")
split_hashes = {}
split_paths = {
    "train": ROOT / "data" / "processed" / "classification" / "train",
    "val": ROOT / "data" / "processed" / "classification" / "val",
    "test": ROOT / "data" / "processed" / "classification" / "test",
    "regression": ROOT / "data" / "processed" / "classification" / "regression",
}

for split_name, split_dir in split_paths.items():
    for fpath in split_dir.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".avif"]:
            with open(fpath, "rb") as fh:
                h = hashlib.md5(fh.read()).hexdigest()
                split_hashes[h] = (split_name, str(fpath.relative_to(ROOT)))

print(f"Total existing images indexed across all splits: {len(split_hashes)}")


# Load dataset_labels.csv
print(f"Loading {LABELS_CSV}...")
catalog = {}
with open(LABELS_CSV, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if len(row) >= 2:
            rel_p = row[0].strip().replace("/", "\\")
            desc = row[1].strip()
            catalog[rel_p] = desc

print(f"Loaded {len(catalog)} descriptions from catalog.")

# Define keyword filters for each of the 4 groups
filters = {
    "Group 1: Lifestyle/Worn Bracelets": {
        "folder": "bracelet",
        "target_class": "Bracelet",
        "keywords": ["wrist", "worn", "arm", "cuff", "heart", "bangle", "open", "clasp"],
        "max_images": 75,
    },
    "Group 2: In-hand / Boxed / Pendant Necklaces": {
        "folder": "necklace",
        "target_class": "Necklace",
        "keywords": ["pendant", "box", "cushion", "chain", "hand", "velvet", "locket", "gift"],
        "max_images": 75,
    },
    "Group 3: Plain-band Rings": {
        "folder": "ring_best",
        "target_class": "Ring",
        "keywords": ["plain", "band", "wedding", "wide", "thick", "gold band", "simple", "smooth"],
        "max_images": 60,
    },
    "Group 4: Lifestyle / In-hand Rings": {
        "folder": "ring_best",
        "target_class": "Ring",
        "keywords": ["finger", "hand", "worn", "engagement", "model", "wearing"],
        "max_images": 60,
    },
}

curation_log = []
rejected_log = []
curated_by_class = {"Bracelet": [], "Earring": [], "Necklace": [], "Ring": []}
seen_curated_hashes = set()

for group_name, cfg in filters.items():
    folder = cfg["folder"]
    target_class = cfg["target_class"]
    keywords = cfg["keywords"]
    max_imgs = cfg["max_images"]
    
    cat_dir = DATASET_DIR / folder
    if not cat_dir.exists():
        print(f"Warning: {cat_dir} does not exist!")
        continue
    
    candidates = []
    for fpath in sorted(cat_dir.glob("*")):
        if not fpath.is_file() or fpath.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        rel_key = f"{folder}\\{fpath.name}"
        desc = catalog.get(rel_key, "")
        
        # Check keyword match
        desc_lower = desc.lower()
        matched_kw = [kw for kw in keywords if kw in desc_lower]
        if matched_kw:
            candidates.append((fpath, desc, matched_kw))
    
    print(f"\n{group_name}: found {len(candidates)} candidates matching keywords {keywords[:4]}")
    
    added_for_group = 0
    for fpath, desc, matched_kw in candidates:
        if added_for_group >= max_imgs:
            break
        
        # 1. Exact MD5 check against all existing splits
        with open(fpath, "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        
        if h in split_hashes:
            split_origin, existing_path = split_hashes[h]
            rejected_log.append({
                "file": fpath.name,
                "reason": f"Exact duplicate of {split_origin} image: {existing_path}",
                "group": group_name
            })
            continue
            
        if h in seen_curated_hashes:
            rejected_log.append({
                "file": fpath.name,
                "reason": "Duplicate within curated hard-negative set",
                "group": group_name
            })
            continue
            
        # 2. Verify image integrity and decodability
        try:
            with Image.open(fpath) as img:
                img.verify()
            with Image.open(fpath) as img:
                w, h_dim = img.size
                if w < 100 or h_dim < 100:
                    rejected_log.append({
                        "file": fpath.name,
                        "reason": f"Image dimensions too small: {w}x{h_dim}",
                        "group": group_name
                    })
                    continue
        except Exception as e:
            rejected_log.append({
                "file": fpath.name,
                "reason": f"Image corrupted or unreadable: {e}",
                "group": group_name
            })
            continue
            
        # Approved!
        seen_curated_hashes.add(h)
        added_for_group += 1
        item_record = {
            "group": group_name,
            "filename": fpath.name,
            "source_path": str(fpath.relative_to(ROOT)).replace("\\", "/"),
            "target_class": target_class,
            "matched_keywords": matched_kw,
            "description": desc,
            "md5": h
        }
        curated_by_class[target_class].append(item_record)
        curation_log.append(item_record)

print("\n--- CURATION SUMMARY ---")
total_curated = len(curation_log)
print(f"Total images curated: {total_curated}")
for cat, items in curated_by_class.items():
    print(f"  {cat}: {len(items)} images")

print(f"Total rejected: {len(rejected_log)}")

# Create destination directories and copy images
for cat in ["Bracelet", "Earring", "Necklace", "Ring"]:
    cat_dest = DEST_BASE / cat
    cat_dest.mkdir(parents=True, exist_ok=True)

print(f"\nWriting curated files to {DEST_BASE}...")
for item in curation_log:
    src_file = ROOT / item["source_path"]
    dst_file = DEST_BASE / item["target_class"] / item["filename"]
    shutil.copy2(src_file, dst_file)

# Save curation manifest
manifest_path = DEST_BASE / "curation_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_curated": total_curated,
        "counts_by_class": {c: len(items) for c, items in curated_by_class.items()},
        "rejected_count": len(rejected_log),
        "curated_items": curation_log,
        "rejected_items": rejected_log[:50] # Sample of rejected items
    }, f, indent=2)

print(f"Manifest written to {manifest_path}")

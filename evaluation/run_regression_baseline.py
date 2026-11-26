"""Regression Baseline Evaluation Script for failure_predicted / real-world evaluation candidate set."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
try:
    import pillow_avif
except ImportError:
    pass
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vision.classifier import CATEGORIES, ClassifierService
from app.vision.preprocessing import (
    extract_morphological_features,
    preprocess_for_classification,
)

REGRESSION_DIR = ROOT / "data" / "processed" / "classification" / "regression"
OUT_DIR = ROOT / "experiments" / "regression_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], destination: Path) -> None:
    cell, header = 110, 145
    image = Image.new("RGB", (header + cell * len(labels), header + cell * len(labels)), "white")
    draw = ImageDraw.Draw(image)
    draw.text((header, 8), "predicted", fill="black")
    draw.text((8, header), "actual", fill="black")
    max_value = max(int(cm.max()), 1)
    for i, actual in enumerate(labels):
        draw.text((5, header + i * cell + cell // 2), actual, fill="black")
        for j, predicted in enumerate(labels):
            if i == 0:
                draw.text((header + j * cell + 5, 35), predicted, fill="black")
            value = int(cm[i, j])
            shade = int(255 - 190 * value / max_value)
            x0, y0 = header + j * cell, header + i * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=(shade, shade, 255), outline="black")
            draw.text((x0 + 45, y0 + 45), str(value), fill="black")
    image.save(destination)


def run_regression_eval():
    service = ClassifierService(device="cuda" if torch.cuda.is_available() else "cpu")
    model, device = service.model, service.device
    model.eval()

    samples: list[tuple[Path, str, int]] = []
    for cat_idx, category in enumerate(CATEGORIES):
        cat_dir = REGRESSION_DIR / category
        if cat_dir.exists():
            for p in sorted(cat_dir.glob("*")):
                if p.is_file():
                    samples.append((p, category, cat_idx))

    if not samples:
        raise FileNotFoundError(f"No regression images found in {REGRESSION_DIR}")

    print(f"Loaded {len(samples)} regression test images across {len(CATEGORIES)} categories.")

    rows = []
    failure_rows = []

    with torch.no_grad():
        for path, gt_class, gt_idx in samples:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            
            tensor = preprocess_for_classification(rgb).to(device)
            morph = extract_morphological_features(rgb)
            morph_tensor = torch.from_numpy(morph).float().unsqueeze(0).to(device)

            logits = model(tensor, morph_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            order = np.argsort(probs)[::-1]
            top1_idx = int(order[0])
            top2_idx = int(order[1])
            pred_class = CATEGORIES[top1_idx]
            confidence = float(probs[top1_idx])
            top2_prob = float(probs[top2_idx])
            margin = float(confidence - top2_prob)
            is_correct = bool(pred_class == gt_class)

            rel_path = str(path.relative_to(ROOT)).replace("\\", "/")

            row = {
                "image_path": rel_path,
                "ground_truth_class": gt_class,
                "predicted_class": pred_class,
                "confidence": confidence,
                "second_highest_probability": top2_prob,
                "probability_margin": margin,
                "correct": is_correct,
                "P_Bracelet": float(probs[0]),
                "P_Earring": float(probs[1]),
                "P_Necklace": float(probs[2]),
                "P_Ring": float(probs[3]),
                "morph_circularity": float(morph[0]),
                "morph_aspect_ratio": float(morph[1]),
                "morph_extent": float(morph[2]),
                "morph_solidity": float(morph[3]),
            }
            rows.append(row)
            if not is_correct:
                failure_rows.append(row)

    # Calculate metrics
    y_true = [CATEGORIES.index(r["ground_truth_class"]) for r in rows]
    y_pred = [CATEGORIES.index(r["predicted_class"]) for r in rows]

    report_dict = classification_report(y_true, y_pred, target_names=CATEGORIES, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=CATEGORIES, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CATEGORIES)))

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    macro_prec = float(report_dict["macro avg"]["precision"])
    macro_rec = float(report_dict["macro avg"]["recall"])

    # 1. Save predictions.csv
    pred_csv = OUT_DIR / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {pred_csv}")

    # 2. Save failures.csv
    fail_csv = OUT_DIR / "failures.csv"
    if failure_rows:
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failure_rows)
    else:
        with open(fail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
    print(f"Saved {fail_csv} ({len(failure_rows)} failures)")

    # 3. Save classification_report.txt
    report_path = OUT_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Regression Baseline Classification Report ===\n\n")
        f.write(report_text)
        f.write("\n\nConfusion Matrix (rows=actual, columns=predicted):\n")
        f.write("Order: " + ", ".join(CATEGORIES) + "\n")
        f.write(np.array2string(cm) + "\n")
    print(f"Saved {report_path}")

    # 4. Save confusion_matrix.png
    cm_path = OUT_DIR / "confusion_matrix.png"
    _save_confusion_matrix(cm, CATEGORIES, cm_path)
    print(f"Saved {cm_path}")

    # Group failures by 12 pairs
    pair_counts = {}
    pair_details = {}
    for c1 in CATEGORIES:
        for c2 in CATEGORIES:
            if c1 != c2:
                pair_name = f"{c1} -> {c2}"
                pair_counts[pair_name] = 0
                pair_details[pair_name] = []

    for f_row in failure_rows:
        pair_name = f"{f_row['ground_truth_class']} -> {f_row['predicted_class']}"
        pair_counts[pair_name] += 1
        pair_details[pair_name].append({
            "image": f_row["image_path"],
            "confidence": f_row["confidence"],
            "second_highest": f_row["second_highest_probability"],
            "margin": f_row["probability_margin"]
        })

    # Per-class counts
    per_class_counts = {cat: sum(1 for r in rows if r["ground_truth_class"] == cat) for cat in CATEGORIES}

    # 5. Save metrics.json
    metrics_data = {
        "evaluation_name": "regression_baseline",
        "dataset_path": str(REGRESSION_DIR.relative_to(ROOT)).replace("\\", "/"),
        "total_images": len(rows),
        "per_class_counts": per_class_counts,
        "accuracy": accuracy,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class_metrics": {cat: report_dict[cat] for cat in CATEGORIES},
        "confusion_matrix": cm.tolist(),
        "categories": CATEGORIES,
        "failure_count": len(failure_rows),
        "confusion_pairs": pair_counts,
        "failure_details": pair_details,
    }
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved {metrics_path}")

    return metrics_data, rows, failure_rows


if __name__ == "__main__":
    run_regression_eval()

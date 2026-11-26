import os
import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import pillow_avif
except ImportError:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vision.classifier import CATEGORIES, JewelleryClassifier
from app.vision.preprocessing import (
    preprocess_for_classification,
    extract_morphological_features,
)

B0_PATH = ROOT / "indexes" / "classifier_resnet50.pth"
B1_PATH = ROOT / "indexes" / "classifier_resnet50_b1.pth"
TEST_DIR = ROOT / "data" / "processed" / "classification" / "test"
REGRESSION_DIR = ROOT / "data" / "processed" / "classification" / "regression"
OUT_DIR = ROOT / "experiments" / "B1_evaluation"


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


def load_model(checkpoint_path: Path, device: torch.device) -> JewelleryClassifier:
    model = JewelleryClassifier(num_classes=len(CATEGORIES), pretrained=False, backbone_name="resnet50")
    state_dict = torch.load(checkpoint_path, map_location=device)
    remapped_sd = {}
    for k, v in state_dict.items():
        if k == "backbone.fc.weight":
            remapped_sd["backbone.fc.1.weight"] = v
        elif k == "backbone.fc.bias":
            remapped_sd["backbone.fc.1.bias"] = v
        else:
            remapped_sd[k] = v
    try:
        model.load_state_dict(remapped_sd)
    except Exception:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_dataset(
    model: JewelleryClassifier,
    dataset_dir: Path,
    device: torch.device,
    is_regression: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    samples: List[Tuple[Path, str, int]] = []
    for cat_idx, cat in enumerate(CATEGORIES):
        cat_dir = dataset_dir / cat
        if cat_dir.exists():
            for p in sorted(cat_dir.glob("*")):
                if p.is_file():
                    samples.append((p, cat, cat_idx))

    rows = []
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
                "filename": path.name,
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
            }
            rows.append(row)

    y_true = [CATEGORIES.index(r["ground_truth_class"]) for r in rows]
    y_pred = [CATEGORIES.index(r["predicted_class"]) for r in rows]

    report_dict = classification_report(y_true, y_pred, target_names=CATEGORIES, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=CATEGORIES, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CATEGORIES)))

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    macro_prec = float(report_dict["macro avg"]["precision"])
    macro_rec = float(report_dict["macro avg"]["recall"])

    # Specific confusion pairs
    pair_counts = {}
    for i, c1 in enumerate(CATEGORIES):
        for j, c2 in enumerate(CATEGORIES):
            if c1 != c2:
                pair_counts[f"{c1} -> {c2}"] = int(cm[i, j])

    metrics = {
        "total_images": len(rows),
        "accuracy": accuracy,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class": {
            cat: {
                "precision": float(report_dict[cat]["precision"]),
                "recall": float(report_dict[cat]["recall"]),
                "f1-score": float(report_dict[cat]["f1-score"]),
                "support": int(report_dict[cat]["support"]),
            }
            for cat in CATEGORIES
        },
        "confusion_matrix": cm.tolist(),
        "confusion_pairs": pair_counts,
        "report_text": report_text,
    }

    return metrics, rows


def run_full_evaluation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}", flush=True)

    print(f"Loading B0 from {B0_PATH}...", flush=True)
    b0_model = load_model(B0_PATH, device)

    print(f"Loading B1 from {B1_PATH}...", flush=True)
    b1_model = load_model(B1_PATH, device)

    # 1. Evaluate on 393-image Test Set
    print("\n--- Evaluating B0 on 393 Test Images ---", flush=True)
    b0_test_metrics, b0_test_rows = evaluate_dataset(b0_model, TEST_DIR, device)

    print("--- Evaluating B1 on 393 Test Images ---", flush=True)
    b1_test_metrics, b1_test_rows = evaluate_dataset(b1_model, TEST_DIR, device)

    # 2. Evaluate on 10-image Regression Set
    print("\n--- Evaluating B0 on 10 Regression Images ---", flush=True)
    b0_reg_metrics, b0_reg_rows = evaluate_dataset(b0_model, REGRESSION_DIR, device, is_regression=True)

    print("--- Evaluating B1 on 10 Regression Images ---", flush=True)
    b1_reg_metrics, b1_reg_rows = evaluate_dataset(b1_model, REGRESSION_DIR, device, is_regression=True)

    # Save test predictions
    for name, rows, cm, metrics in [
        ("b0_test", b0_test_rows, b0_test_metrics["confusion_matrix"], b0_test_metrics),
        ("b1_test", b1_test_rows, b1_test_metrics["confusion_matrix"], b1_test_metrics),
    ]:
        with open(OUT_DIR / f"{name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        _save_confusion_matrix(np.array(cm), CATEGORIES, OUT_DIR / f"{name}_confusion_matrix.png")

    # 3. Regression detailed comparison table (Part C)
    # Known failure descriptions mapped to images:
    known_failure_mapping = {
        "sample bracelet.webp": "Bracelet photographed on wrist/lifestyle background -> Earring",
        "test b.jpg": "Open cuff bracelet -> Earring",
        "sample chain.webp": "Necklace held across hands -> Earring",
        "vrace.jpg": "Necklace in gift box with prominent pendant -> Earring",
        "ring1.jpg": "Plain gold band without scale context -> Bracelet",
        "sample ring.webp": "Ring in lifestyle/hand context -> Earring",
    }

    reg_table = []
    b0_reg_map = {r["filename"]: r for r in b0_reg_rows}
    b1_reg_map = {r["filename"]: r for r in b1_reg_rows}

    # Sort filenames consistently
    all_reg_files = sorted(b0_reg_map.keys())

    fixed_count = 0
    new_error_count = 0

    for fname in all_reg_files:
        r0 = b0_reg_map[fname]
        r1 = b1_reg_map[fname]

        gt = r0["ground_truth_class"]
        p0 = r0["predicted_class"]
        c0 = r0["confidence"]
        p1 = r1["predicted_class"]
        c1 = r1["confidence"]

        b0_wrong = not r0["correct"]
        b1_wrong = not r1["correct"]

        b1_fixed = b0_wrong and (not b1_wrong)
        b1_new_error = (not b0_wrong) and b1_wrong

        if b1_fixed:
            fixed_count += 1
        if b1_new_error:
            new_error_count += 1

        known_failure_desc = known_failure_mapping.get(fname, "N/A (Correct in B0)")

        reg_table.append({
            "filename": fname,
            "true_class": gt,
            "b0_prediction": p0,
            "b0_confidence": c0,
            "b1_prediction": p1,
            "b1_confidence": c1,
            "b0_wrong": b0_wrong,
            "b1_fixed_error": b1_fixed,
            "b1_new_error": b1_new_error,
            "known_failure_description": known_failure_desc,
        })

    # Save regression comparison CSV
    with open(OUT_DIR / "regression_comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(reg_table[0].keys()))
        writer.writeheader()
        writer.writerows(reg_table)

    with open(OUT_DIR / "regression_comparison_table.json", "w", encoding="utf-8") as f:
        json.dump(reg_table, f, indent=2)

    # 4. Save combined metrics comparison JSON
    comparison = {
        "test_set": {
            "total_images": 393,
            "b0": b0_test_metrics,
            "b1": b1_test_metrics,
        },
        "regression_set": {
            "total_images": 10,
            "b0": b0_reg_metrics,
            "b1": b1_reg_metrics,
            "fixed_count": fixed_count,
            "new_error_count": new_error_count,
            "detailed_table": reg_table,
        },
    }

    with open(OUT_DIR / "metrics_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # 5. Generate comprehensive textual report
    report_lines = []
    report_lines.append("================================================================================")
    report_lines.append("                   JEWELLERY CLASSIFIER B0 vs B1 EVALUATION REPORT              ")
    report_lines.append("================================================================================")
    report_lines.append(f"B0 Checkpoint: {B0_PATH}")
    report_lines.append(f"B1 Checkpoint: {B1_PATH}\n")

    report_lines.append("--------------------------------------------------------------------------------")
    report_lines.append("PART 1: 393-IMAGE TEST SET COMPARISON")
    report_lines.append("--------------------------------------------------------------------------------")
    report_lines.append(f"{'Metric':<30} | {'B0':<12} | {'B1':<12} | {'Delta':<12}")
    report_lines.append("-" * 72)
    acc0 = b0_test_metrics['accuracy'] * 100
    acc1 = b1_test_metrics['accuracy'] * 100
    report_lines.append(f"{'Test Accuracy':<30} | {acc0:6.2f}%      | {acc1:6.2f}%      | {acc1 - acc0:+6.2f}%")

    f1_0 = b0_test_metrics['macro_f1']
    f1_1 = b1_test_metrics['macro_f1']
    report_lines.append(f"{'Macro-F1':<30} | {f1_0:8.4f}     | {f1_1:8.4f}     | {f1_1 - f1_0:+8.4f}")

    p0 = b0_test_metrics['macro_precision']
    p1 = b1_test_metrics['macro_precision']
    report_lines.append(f"{'Macro-Precision':<30} | {p0:8.4f}     | {p1:8.4f}     | {p1 - p0:+8.4f}")

    r0 = b0_test_metrics['macro_recall']
    r1 = b1_test_metrics['macro_recall']
    report_lines.append(f"{'Macro-Recall':<30} | {r0:8.4f}     | {r1:8.4f}     | {r1 - r0:+8.4f}")
    report_lines.append("")

    report_lines.append("--- Per-Class Metrics (Precision / Recall / F1) ---")
    for cat in CATEGORIES:
        c0 = b0_test_metrics['per_class'][cat]
        c1 = b1_test_metrics['per_class'][cat]
        report_lines.append(
            f"{cat:<10} | B0: P={c0['precision']:.4f}, R={c0['recall']:.4f}, F1={c0['f1-score']:.4f} "
            f"| B1: P={c1['precision']:.4f}, R={c1['recall']:.4f}, F1={c1['f1-score']:.4f}"
        )
    report_lines.append("")

    report_lines.append("--- Key Confusion Pairs ---")
    for pair in [
        "Necklace -> Earring",
        "Necklace -> Bracelet",
        "Bracelet -> Earring",
        "Ring -> Bracelet",
        "Ring -> Earring",
        "Bracelet -> Ring",
        "Earring -> Bracelet",
        "Earring -> Ring",
    ]:
        v0 = b0_test_metrics['confusion_pairs'].get(pair, 0)
        v1 = b1_test_metrics['confusion_pairs'].get(pair, 0)
        report_lines.append(f"{pair:<25}: B0 = {v0:2d}  |  B1 = {v1:2d}  (Delta: {v1 - v0:+2d})")
    report_lines.append("")

    report_lines.append("B0 Confusion Matrix (Bracelet, Earring, Necklace, Ring):")
    report_lines.append(np.array2string(np.array(b0_test_metrics['confusion_matrix'])))
    report_lines.append("\nB1 Confusion Matrix (Bracelet, Earring, Necklace, Ring):")
    report_lines.append(np.array2string(np.array(b1_test_metrics['confusion_matrix'])))
    report_lines.append("")

    report_lines.append("--------------------------------------------------------------------------------")
    report_lines.append("PART 2: 10-IMAGE REAL-WORLD REGRESSION COMPARISON")
    report_lines.append("--------------------------------------------------------------------------------")
    reg_acc0 = b0_reg_metrics['accuracy'] * 100
    reg_acc1 = b1_reg_metrics['accuracy'] * 100
    report_lines.append(f"Real-World Regression Accuracy: B0 = {reg_acc0:.1f}% ({int(b0_reg_metrics['accuracy']*10)}/10) | B1 = {reg_acc1:.1f}% ({int(b1_reg_metrics['accuracy']*10)}/10)\n")

    report_lines.append(f"{'Filename':<35} | {'True':<9} | {'B0 Pred':<9} | {'B0 Conf':<7} | {'B1 Pred':<9} | {'B1 Conf':<7} | {'B0 Wrong':<8} | {'B1 Fixed':<8} | {'New Err':<7}")
    report_lines.append("-" * 115)
    for r in reg_table:
        report_lines.append(
            f"{r['filename']:<35} | {r['true_class']:<9} | {r['b0_prediction']:<9} | {r['b0_confidence']:6.3f}  | "
            f"{r['b1_prediction']:<9} | {r['b1_confidence']:6.3f}  | {str(r['b0_wrong']):<8} | {str(r['b1_fixed_error']):<8} | {str(r['b1_new_error']):<7}"
        )
    report_lines.append("")

    report_lines.append("--- Six Known Real-World Failures Analysis ---")
    for r in reg_table:
        if r["known_failure_description"] != "N/A (Correct in B0)":
            status = "FIXED" if r["b1_fixed_error"] else ("STILL FAILING" if (not r["b1_fixed_error"] and r["b0_wrong"]) else "N/A")
            report_lines.append(f"- Pattern: {r['known_failure_description']}")
            report_lines.append(f"  File: {r['filename']} (True: {r['true_class']})")
            report_lines.append(f"  B0: {r['b0_prediction']} ({r['b0_confidence']:.3f}) -> B1: {r['b1_prediction']} ({r['b1_confidence']:.3f}) [{status}]")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    with open(OUT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    return comparison


if __name__ == "__main__":
    run_full_evaluation()

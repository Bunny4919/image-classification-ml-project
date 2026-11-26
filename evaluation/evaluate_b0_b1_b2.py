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


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.vision.classifier import CATEGORIES, JewelleryClassifier
from app.vision.preprocessing import (
    preprocess_for_classification,
    extract_morphological_features,
)


# ============================================================
# CHECKPOINTS / DATA / OUTPUT
# ============================================================

B0_PATH = ROOT / "indexes" / "classifier_resnet50.pth"
B1_PATH = ROOT / "indexes" / "classifier_resnet50_b1.pth"
B2_PATH = ROOT / "indexes" / "classifier_resnet50_b2.pth"

TEST_DIR = ROOT / "data" / "processed" / "classification" / "test"
REGRESSION_DIR = ROOT / "data" / "processed" / "classification" / "regression"

OUT_DIR = ROOT / "experiments" / "B2_evaluation"


# ============================================================
# CONFUSION MATRIX IMAGE
# ============================================================

def _save_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    destination: Path,
) -> None:

    cell, header = 110, 145

    image = Image.new(
        "RGB",
        (
            header + cell * len(labels),
            header + cell * len(labels),
        ),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.text((header, 8), "predicted", fill="black")
    draw.text((8, header), "actual", fill="black")

    max_value = max(int(cm.max()), 1)

    for i, actual in enumerate(labels):

        draw.text(
            (
                5,
                header + i * cell + cell // 2,
            ),
            actual,
            fill="black",
        )

        for j, predicted in enumerate(labels):

            if i == 0:
                draw.text(
                    (
                        header + j * cell + 5,
                        35,
                    ),
                    predicted,
                    fill="black",
                )

            value = int(cm[i, j])

            shade = int(
                255 - 190 * value / max_value
            )

            x0 = header + j * cell
            y0 = header + i * cell

            draw.rectangle(
                (
                    x0,
                    y0,
                    x0 + cell - 2,
                    y0 + cell - 2,
                ),
                fill=(shade, shade, 255),
                outline="black",
            )

            draw.text(
                (
                    x0 + 45,
                    y0 + 45,
                ),
                str(value),
                fill="black",
            )

    image.save(destination)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> JewelleryClassifier:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print(
        f"Loading checkpoint: {checkpoint_path}",
        flush=True,
    )

    model = JewelleryClassifier(
        num_classes=len(CATEGORIES),
        pretrained=False,
        backbone_name="resnet50",
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Handle older checkpoint naming:
    # backbone.fc.weight -> backbone.fc.1.weight
    # backbone.fc.bias   -> backbone.fc.1.bias

    remapped_sd = {}

    for k, v in state_dict.items():

        if k == "backbone.fc.weight":

            remapped_sd[
                "backbone.fc.1.weight"
            ] = v

        elif k == "backbone.fc.bias":

            remapped_sd[
                "backbone.fc.1.bias"
            ] = v

        else:

            remapped_sd[k] = v

    try:

        model.load_state_dict(remapped_sd)

    except Exception:

        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# ============================================================
# EVALUATE ONE DATASET
# ============================================================

def evaluate_dataset(
    model: JewelleryClassifier,
    dataset_dir: Path,
    device: torch.device,
    is_regression: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    samples: List[
        Tuple[Path, str, int]
    ] = []

    # --------------------------------------------------------
    # Collect samples
    # --------------------------------------------------------

    for cat_idx, cat in enumerate(CATEGORIES):

        cat_dir = dataset_dir / cat

        if cat_dir.exists():

            for p in sorted(cat_dir.glob("*")):

                if p.is_file():

                    samples.append(
                        (
                            p,
                            cat,
                            cat_idx,
                        )
                    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    rows = []

    with torch.no_grad():

        for path, gt_class, gt_idx in samples:

            with Image.open(path) as img:

                rgb = img.convert("RGB")

            tensor = preprocess_for_classification(
                rgb
            ).to(device)

            morph = extract_morphological_features(
                rgb
            )

            morph_tensor = (
                torch.from_numpy(morph)
                .float()
                .unsqueeze(0)
                .to(device)
            )

            logits = model(
                tensor,
                morph_tensor,
            )

            probs = (
                F.softmax(
                    logits,
                    dim=1,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Top predictions
            # ------------------------------------------------

            order = np.argsort(probs)[::-1]

            top1_idx = int(order[0])
            top2_idx = int(order[1])

            pred_class = CATEGORIES[top1_idx]

            confidence = float(
                probs[top1_idx]
            )

            top2_prob = float(
                probs[top2_idx]
            )

            margin = float(
                confidence - top2_prob
            )

            is_correct = bool(
                pred_class == gt_class
            )

            rel_path = str(
                path.relative_to(ROOT)
            ).replace("\\", "/")

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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    y_true = [
        CATEGORIES.index(
            r["ground_truth_class"]
        )
        for r in rows
    ]

    y_pred = [
        CATEGORIES.index(
            r["predicted_class"]
        )
        for r in rows
    ]

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=CATEGORIES,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(CATEGORIES)),
    )

    accuracy = float(
        np.mean(
            np.array(y_true)
            == np.array(y_pred)
        )
    )

    macro_f1 = float(
        report_dict["macro avg"]["f1-score"]
    )

    macro_prec = float(
        report_dict["macro avg"]["precision"]
    )

    macro_rec = float(
        report_dict["macro avg"]["recall"]
    )

    # --------------------------------------------------------
    # Specific confusion pairs
    # --------------------------------------------------------

    pair_counts = {}

    for i, c1 in enumerate(CATEGORIES):

        for j, c2 in enumerate(CATEGORIES):

            if c1 != c2:

                pair_counts[
                    f"{c1} -> {c2}"
                ] = int(cm[i, j])

    # --------------------------------------------------------
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {

        "total_images": len(rows),

        "accuracy": accuracy,

        "macro_precision": macro_prec,

        "macro_recall": macro_rec,

        "macro_f1": macro_f1,

        "per_class": {

            cat: {

                "precision": float(
                    report_dict[cat]["precision"]
                ),

                "recall": float(
                    report_dict[cat]["recall"]
                ),

                "f1-score": float(
                    report_dict[cat]["f1-score"]
                ),

                "support": int(
                    report_dict[cat]["support"]
                ),
            }

            for cat in CATEGORIES
        },

        "confusion_matrix": cm.tolist(),

        "confusion_pairs": pair_counts,

        "report_text": report_text,
    }

    return metrics, rows


# ============================================================
# REGRESSION COMPARISON
# ============================================================

def build_regression_table(
    b0_rows,
    b1_rows,
    b2_rows,
):

    known_failure_mapping = {

        "sample bracelet.webp":
            "Bracelet photographed on wrist/lifestyle background -> Earring",

        "test b.jpg":
            "Open cuff bracelet -> Earring",

        "sample chain.webp":
            "Necklace held across hands -> Earring",

        "vrace.jpg":
            "Necklace in gift box with prominent pendant -> Earring",

        "ring1.jpg":
            "Plain gold band without scale context -> Bracelet",

        "sample ring.webp":
            "Ring in lifestyle/hand context -> Earring",
    }

    b0_map = {
        r["filename"]: r
        for r in b0_rows
    }

    b1_map = {
        r["filename"]: r
        for r in b1_rows
    }

    b2_map = {
        r["filename"]: r
        for r in b2_rows
    }

    all_reg_files = sorted(
        set(b0_map.keys())
        | set(b1_map.keys())
        | set(b2_map.keys())
    )

    b1_fixed_count = 0
    b2_fixed_count = 0

    b1_new_error_count = 0
    b2_new_error_count = 0

    reg_table = []

    for fname in all_reg_files:

        r0 = b0_map[fname]
        r1 = b1_map[fname]
        r2 = b2_map[fname]

        gt = r0["ground_truth_class"]

        p0 = r0["predicted_class"]
        p1 = r1["predicted_class"]
        p2 = r2["predicted_class"]

        c0 = r0["confidence"]
        c1 = r1["confidence"]
        c2 = r2["confidence"]

        b0_wrong = not r0["correct"]
        b1_wrong = not r1["correct"]
        b2_wrong = not r2["correct"]

        # ----------------------------------------------------
        # B1 relative to B0
        # ----------------------------------------------------

        b1_fixed = (
            b0_wrong
            and not b1_wrong
        )

        b1_new_error = (
            not b0_wrong
            and b1_wrong
        )

        # ----------------------------------------------------
        # B2 relative to B0
        # ----------------------------------------------------

        b2_fixed_vs_b0 = (
            b0_wrong
            and not b2_wrong
        )

        b2_new_error_vs_b0 = (
            not b0_wrong
            and b2_wrong
        )

        # ----------------------------------------------------
        # B2 relative to B1
        # ----------------------------------------------------

        b2_fixed_vs_b1 = (
            b1_wrong
            and not b2_wrong
        )

        b2_new_error_vs_b1 = (
            not b1_wrong
            and b2_wrong
        )

        if b1_fixed:
            b1_fixed_count += 1

        if b2_fixed_vs_b0:
            b2_fixed_count += 1

        if b1_new_error:
            b1_new_error_count += 1

        if b2_new_error_vs_b0:
            b2_new_error_count += 1

        known_failure_desc = known_failure_mapping.get(
            fname,
            "N/A (Correct in B0)",
        )

        reg_table.append({

            "filename": fname,

            "true_class": gt,

            # B0
            "b0_prediction": p0,
            "b0_confidence": c0,
            "b0_wrong": b0_wrong,

            # B1
            "b1_prediction": p1,
            "b1_confidence": c1,
            "b1_wrong": b1_wrong,
            "b1_fixed_vs_b0": b1_fixed,
            "b1_new_error_vs_b0": b1_new_error,

            # B2
            "b2_prediction": p2,
            "b2_confidence": c2,
            "b2_wrong": b2_wrong,

            "b2_fixed_vs_b0": b2_fixed_vs_b0,
            "b2_new_error_vs_b0": b2_new_error_vs_b0,

            "b2_fixed_vs_b1": b2_fixed_vs_b1,
            "b2_new_error_vs_b1": b2_new_error_vs_b1,

            "known_failure_description":
                known_failure_desc,
        })

    return (
        reg_table,
        b1_fixed_count,
        b2_fixed_count,
        b1_new_error_count,
        b2_new_error_count,
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    name: str,
    rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
):

    with open(
        OUT_DIR / f"{name}_predictions.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    _save_confusion_matrix(
        np.array(
            metrics["confusion_matrix"]
        ),
        CATEGORIES,
        OUT_DIR / f"{name}_confusion_matrix.png",
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_full_evaluation():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "JEWELLERY CLASSIFIER B0 / B1 / B2 EVALUATION",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Running evaluation on device: {device}",
        flush=True,
    )

    print(
        f"Test directory: {TEST_DIR}",
        flush=True,
    )

    print(
        f"Regression directory: {REGRESSION_DIR}",
        flush=True,
    )

    print(
        f"Output directory: {OUT_DIR}",
        flush=True,
    )

    # ========================================================
    # Verify checkpoints exist
    # ========================================================

    for checkpoint_name, checkpoint_path in [
        ("B0", B0_PATH),
        ("B1", B1_PATH),
        ("B2", B2_PATH),
    ]:

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"{checkpoint_name} checkpoint not found:\n"
                f"{checkpoint_path}"
            )

        print(
            f"{checkpoint_name} checkpoint found.",
            flush=True,
        )

    # ========================================================
    # Load models
    # ========================================================

    print(
        "\n--- Loading B0 ---",
        flush=True,
    )

    b0_model = load_model(
        B0_PATH,
        device,
    )

    print(
        "\n--- Loading B1 ---",
        flush=True,
    )

    b1_model = load_model(
        B1_PATH,
        device,
    )

    print(
        "\n--- Loading B2 ---",
        flush=True,
    )

    b2_model = load_model(
        B2_PATH,
        device,
    )

    # ========================================================
    # TEST SET
    # ========================================================

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        "PART 1 — 393 IMAGE TEST SET",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "\n--- Evaluating B0 on Test ---",
        flush=True,
    )

    b0_test_metrics, b0_test_rows = evaluate_dataset(
        b0_model,
        TEST_DIR,
        device,
    )

    print(
        "\n--- Evaluating B1 on Test ---",
        flush=True,
    )

    b1_test_metrics, b1_test_rows = evaluate_dataset(
        b1_model,
        TEST_DIR,
        device,
    )

    print(
        "\n--- Evaluating B2 on Test ---",
        flush=True,
    )

    b2_test_metrics, b2_test_rows = evaluate_dataset(
        b2_model,
        TEST_DIR,
        device,
    )

    # ========================================================
    # REGRESSION SET
    # ========================================================

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        "PART 2 — 10 IMAGE REAL-WORLD REGRESSION",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "\n--- Evaluating B0 on Regression ---",
        flush=True,
    )

    b0_reg_metrics, b0_reg_rows = evaluate_dataset(
        b0_model,
        REGRESSION_DIR,
        device,
        is_regression=True,
    )

    print(
        "\n--- Evaluating B1 on Regression ---",
        flush=True,
    )

    b1_reg_metrics, b1_reg_rows = evaluate_dataset(
        b1_model,
        REGRESSION_DIR,
        device,
        is_regression=True,
    )

    print(
        "\n--- Evaluating B2 on Regression ---",
        flush=True,
    )

    b2_reg_metrics, b2_reg_rows = evaluate_dataset(
        b2_model,
        REGRESSION_DIR,
        device,
        is_regression=True,
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    save_predictions(
        "b0_test",
        b0_test_rows,
        b0_test_metrics,
    )

    save_predictions(
        "b1_test",
        b1_test_rows,
        b1_test_metrics,
    )

    save_predictions(
        "b2_test",
        b2_test_rows,
        b2_test_metrics,
    )

    # ========================================================
    # REGRESSION TABLE
    # ========================================================

    (
        reg_table,
        b1_fixed_count,
        b2_fixed_count,
        b1_new_error_count,
        b2_new_error_count,
    ) = build_regression_table(
        b0_reg_rows,
        b1_reg_rows,
        b2_reg_rows,
    )

    # ========================================================
    # SAVE REGRESSION CSV
    # ========================================================

    with open(
        OUT_DIR / "regression_comparison_table.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                reg_table[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(reg_table)

    # ========================================================
    # SAVE REGRESSION JSON
    # ========================================================

    with open(
        OUT_DIR / "regression_comparison_table.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            reg_table,
            f,
            indent=2,
        )

    # ========================================================
    # COMBINED METRICS JSON
    # ========================================================

    comparison = {

        "evaluation_info": {

            "test_images": len(
                b0_test_rows
            ),

            "regression_images": len(
                b0_reg_rows
            ),

            "categories": CATEGORIES,

            "b0_checkpoint": str(
                B0_PATH
            ),

            "b1_checkpoint": str(
                B1_PATH
            ),

            "b2_checkpoint": str(
                B2_PATH
            ),
        },

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        "test_set": {

            "total_images":
                len(b0_test_rows),

            "b0":
                b0_test_metrics,

            "b1":
                b1_test_metrics,

            "b2":
                b2_test_metrics,
        },

        # ----------------------------------------------------
        # REGRESSION
        # ----------------------------------------------------

        "regression_set": {

            "total_images":
                len(b0_reg_rows),

            "b0":
                b0_reg_metrics,

            "b1":
                b1_reg_metrics,

            "b2":
                b2_reg_metrics,

            "b1_fixed_count_vs_b0":
                b1_fixed_count,

            "b2_fixed_count_vs_b0":
                b2_fixed_count,

            "b1_new_error_count_vs_b0":
                b1_new_error_count,

            "b2_new_error_count_vs_b0":
                b2_new_error_count,

            "detailed_table":
                reg_table,
        },
    }

    with open(
        OUT_DIR / "metrics_comparison.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparison,
            f,
            indent=2,
        )

    # ========================================================
    # TEXT REPORT
    # ========================================================

    report_lines = []

    report_lines.append(
        "=" * 80
    )

    report_lines.append(
        "JEWELLERY CLASSIFIER B0 vs B1 vs B2 EVALUATION REPORT"
    )

    report_lines.append(
        "=" * 80
    )

    report_lines.append(
        f"B0 Checkpoint: {B0_PATH}"
    )

    report_lines.append(
        f"B1 Checkpoint: {B1_PATH}"
    )

    report_lines.append(
        f"B2 Checkpoint: {B2_PATH}\n"
    )

    # ========================================================
    # TEST METRICS
    # ========================================================

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "PART 1: 393-IMAGE TEST SET COMPARISON"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        f"{'Metric':<30} | "
        f"{'B0':<12} | "
        f"{'B1':<12} | "
        f"{'B2':<12}"
    )

    report_lines.append(
        "-" * 75
    )

    # Accuracy

    acc0 = (
        b0_test_metrics["accuracy"]
        * 100
    )

    acc1 = (
        b1_test_metrics["accuracy"]
        * 100
    )

    acc2 = (
        b2_test_metrics["accuracy"]
        * 100
    )

    report_lines.append(
        f"{'Test Accuracy':<30} | "
        f"{acc0:6.2f}%      | "
        f"{acc1:6.2f}%      | "
        f"{acc2:6.2f}%"
    )

    # Macro F1

    f1_0 = b0_test_metrics[
        "macro_f1"
    ]

    f1_1 = b1_test_metrics[
        "macro_f1"
    ]

    f1_2 = b2_test_metrics[
        "macro_f1"
    ]

    report_lines.append(
        f"{'Macro-F1':<30} | "
        f"{f1_0:8.4f}     | "
        f"{f1_1:8.4f}     | "
        f"{f1_2:8.4f}"
    )

    # Precision

    p0 = b0_test_metrics[
        "macro_precision"
    ]

    p1 = b1_test_metrics[
        "macro_precision"
    ]

    p2 = b2_test_metrics[
        "macro_precision"
    ]

    report_lines.append(
        f"{'Macro-Precision':<30} | "
        f"{p0:8.4f}     | "
        f"{p1:8.4f}     | "
        f"{p2:8.4f}"
    )

    # Recall

    r0 = b0_test_metrics[
        "macro_recall"
    ]

    r1 = b1_test_metrics[
        "macro_recall"
    ]

    r2 = b2_test_metrics[
        "macro_recall"
    ]

    report_lines.append(
        f"{'Macro-Recall':<30} | "
        f"{r0:8.4f}     | "
        f"{r1:8.4f}     | "
        f"{r2:8.4f}"
    )

    report_lines.append("")

    # ========================================================
    # PER CLASS
    # ========================================================

    report_lines.append(
        "--- Per-Class Metrics ---"
    )

    report_lines.append(
        "Format: Precision / Recall / F1"
    )

    report_lines.append("")

    for cat in CATEGORIES:

        c0 = b0_test_metrics[
            "per_class"
        ][cat]

        c1 = b1_test_metrics[
            "per_class"
        ][cat]

        c2 = b2_test_metrics[
            "per_class"
        ][cat]

        report_lines.append(

            f"{cat:<10} | "

            f"B0: "
            f"P={c0['precision']:.4f}, "
            f"R={c0['recall']:.4f}, "
            f"F1={c0['f1-score']:.4f} | "

            f"B1: "
            f"P={c1['precision']:.4f}, "
            f"R={c1['recall']:.4f}, "
            f"F1={c1['f1-score']:.4f} | "

            f"B2: "
            f"P={c2['precision']:.4f}, "
            f"R={c2['recall']:.4f}, "
            f"F1={c2['f1-score']:.4f}"
        )

    report_lines.append("")

    # ========================================================
    # KEY CONFUSION PAIRS
    # ========================================================

    report_lines.append(
        "--- Key Confusion Pairs ---"
    )

    key_pairs = [

        "Necklace -> Earring",

        "Necklace -> Bracelet",

        "Bracelet -> Earring",

        "Ring -> Bracelet",

        "Ring -> Earring",

        "Bracelet -> Ring",

        "Bracelet -> Necklace",

        "Earring -> Bracelet",

        "Earring -> Ring",
    ]

    for pair in key_pairs:

        v0 = b0_test_metrics[
            "confusion_pairs"
        ].get(pair, 0)

        v1 = b1_test_metrics[
            "confusion_pairs"
        ].get(pair, 0)

        v2 = b2_test_metrics[
            "confusion_pairs"
        ].get(pair, 0)

        report_lines.append(

            f"{pair:<25}: "
            f"B0 = {v0:2d} | "
            f"B1 = {v1:2d} | "
            f"B2 = {v2:2d}"
        )

    report_lines.append("")

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    report_lines.append(
        "B0 Confusion Matrix "
        "(Bracelet, Earring, Necklace, Ring):"
    )

    report_lines.append(
        np.array2string(
            np.array(
                b0_test_metrics[
                    "confusion_matrix"
                ]
            )
        )
    )

    report_lines.append("")

    report_lines.append(
        "B1 Confusion Matrix "
        "(Bracelet, Earring, Necklace, Ring):"
    )

    report_lines.append(
        np.array2string(
            np.array(
                b1_test_metrics[
                    "confusion_matrix"
                ]
            )
        )
    )

    report_lines.append("")

    report_lines.append(
        "B2 Confusion Matrix "
        "(Bracelet, Earring, Necklace, Ring):"
    )

    report_lines.append(
        np.array2string(
            np.array(
                b2_test_metrics[
                    "confusion_matrix"
                ]
            )
        )
    )

    report_lines.append("")

    # ========================================================
    # REGRESSION
    # ========================================================

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        "PART 2: 10-IMAGE REAL-WORLD REGRESSION COMPARISON"
    )

    report_lines.append(
        "-" * 80
    )

    reg_acc0 = (
        b0_reg_metrics["accuracy"]
        * 100
    )

    reg_acc1 = (
        b1_reg_metrics["accuracy"]
        * 100
    )

    reg_acc2 = (
        b2_reg_metrics["accuracy"]
        * 100
    )

    total_reg = len(
        b0_reg_rows
    )

    correct0 = sum(
        r["correct"]
        for r in b0_reg_rows
    )

    correct1 = sum(
        r["correct"]
        for r in b1_reg_rows
    )

    correct2 = sum(
        r["correct"]
        for r in b2_reg_rows
    )

    report_lines.append(

        "Real-World Regression Accuracy: "

        f"B0 = {reg_acc0:.1f}% "
        f"({correct0}/{total_reg}) | "

        f"B1 = {reg_acc1:.1f}% "
        f"({correct1}/{total_reg}) | "

        f"B2 = {reg_acc2:.1f}% "
        f"({correct2}/{total_reg})"
    )

    report_lines.append("")

    report_lines.append(
        f"{'Filename':<35} | "
        f"{'True':<9} | "
        f"{'B0 Pred':<9} | "
        f"{'B0 Conf':<7} | "
        f"{'B1 Pred':<9} | "
        f"{'B1 Conf':<7} | "
        f"{'B2 Pred':<9} | "
        f"{'B2 Conf':<7}"
    )

    report_lines.append(
        "-" * 125
    )

    for r in reg_table:

        report_lines.append(

            f"{r['filename']:<35} | "
            f"{r['true_class']:<9} | "

            f"{r['b0_prediction']:<9} | "
            f"{r['b0_confidence']:6.3f}  | "

            f"{r['b1_prediction']:<9} | "
            f"{r['b1_confidence']:6.3f}  | "

            f"{r['b2_prediction']:<9} | "
            f"{r['b2_confidence']:6.3f}"
        )

    report_lines.append("")

    # ========================================================
    # REGRESSION STATUS
    # ========================================================

    report_lines.append(
        "--- Regression Changes ---"
    )

    report_lines.append(
        f"B1 fixed B0 failures: "
        f"{b1_fixed_count}"
    )

    report_lines.append(
        f"B2 fixed B0 failures: "
        f"{b2_fixed_count}"
    )

    report_lines.append(
        f"B1 new errors vs B0: "
        f"{b1_new_error_count}"
    )

    report_lines.append(
        f"B2 new errors vs B0: "
        f"{b2_new_error_count}"
    )

    report_lines.append("")

    # ========================================================
    # SIX KNOWN FAILURE ANALYSIS
    # ========================================================

    report_lines.append(
        "--- Six Known Real-World Failure Analysis ---"
    )

    for r in reg_table:

        if (
            r["known_failure_description"]
            != "N/A (Correct in B0)"
        ):

            b0_status = (
                "CORRECT"
                if not r["b0_wrong"]
                else "WRONG"
            )

            b1_status = (
                "CORRECT"
                if not r["b1_wrong"]
                else "WRONG"
            )

            b2_status = (
                "CORRECT"
                if not r["b2_wrong"]
                else "WRONG"
            )

            report_lines.append(
                f"- Pattern: "
                f"{r['known_failure_description']}"
            )

            report_lines.append(
                f"  File: "
                f"{r['filename']} "
                f"(True: {r['true_class']})"
            )

            report_lines.append(

                f"  B0: "
                f"{r['b0_prediction']} "
                f"({r['b0_confidence']:.3f}) "
                f"[{b0_status}]"
            )

            report_lines.append(

                f"  B1: "
                f"{r['b1_prediction']} "
                f"({r['b1_confidence']:.3f}) "
                f"[{b1_status}]"
            )

            report_lines.append(

                f"  B2: "
                f"{r['b2_prediction']} "
                f"({r['b2_confidence']:.3f}) "
                f"[{b2_status}]"
            )

            report_lines.append("")

    # ========================================================
    # B2 vs B1 CHANGES
    # ========================================================

    b2_fixed_vs_b1_count = sum(
        1
        for r in reg_table
        if r["b2_fixed_vs_b1"]
    )

    b2_new_vs_b1_count = sum(
        1
        for r in reg_table
        if r["b2_new_error_vs_b1"]
    )

    report_lines.append(
        "--- B2 vs B1 Regression Changes ---"
    )

    report_lines.append(
        f"B2 fixed B1 failures: "
        f"{b2_fixed_vs_b1_count}"
    )

    report_lines.append(
        f"B2 introduced errors on B1-correct images: "
        f"{b2_new_vs_b1_count}"
    )

    report_lines.append("")

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    report_text = "\n".join(
        report_lines
    )

    with open(
        OUT_DIR / "evaluation_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report_text)

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        "EVALUATION COMPLETE",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        report_text,
        flush=True,
    )

    print(
        "\nArtifacts saved to:",
        flush=True,
    )

    print(
        OUT_DIR,
        flush=True,
    )

    return comparison


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_full_evaluation()
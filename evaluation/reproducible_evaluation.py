"""Deterministic, file-backed evaluation for the category classifier.

This runner deliberately loads each source image as RGB, derives morphology from
that original image, and calls the same square-pad/normalise preprocessing path
as ``ClassifierService.predict``.  It does not exercise API multi-view crops;
those are an application-level policy and must be evaluated separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
try:
    import pillow_avif
except ImportError:
    pass
from sklearn.metrics import classification_report, confusion_matrix

from app.vision.classifier import CATEGORIES, ClassifierService
from app.vision.preprocessing import (
    extract_morphological_features,
    preprocess_for_classification,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DIR = BASE_DIR / "data" / "processed" / "classification" / "test"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], destination: Path) -> None:
    """Write a dependency-light confusion-matrix PNG."""
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


def evaluate(test_dir: Path, output_dir: Path, batch_size: int = 8, seed: int = 42, zero_morph: bool = False) -> dict:
    set_seed(seed)
    # Small CPU batches avoid the memory spike in the legacy 32-image evaluator.
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[tuple[Path, int]] = []
    for class_index, category in enumerate(CATEGORIES):
        samples.extend((path, class_index) for path in sorted((test_dir / category).glob("*")) if path.is_file())
    if not samples:
        raise FileNotFoundError(f"No evaluation images found under {test_dir}")

    service = ClassifierService(device="cuda" if torch.cuda.is_available() else "cpu")
    model, device = service.model, service.device
    model.eval()
    rows: list[dict] = []

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            group = samples[start:start + batch_size]
            tensors, morphs = [], []
            for path, _ in group:
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                tensors.append(preprocess_for_classification(image).squeeze(0))
                morphs.append(np.zeros(4, dtype=np.float32) if zero_morph else extract_morphological_features(image))
            logits = model(
                torch.stack(tensors).to(device),
                torch.tensor(np.asarray(morphs), dtype=torch.float32, device=device),
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for (path, target), morph, probs in zip(group, morphs, probabilities):
                ordered = np.argsort(probs)[::-1]
                top1, top2 = int(ordered[0]), int(ordered[1])
                rows.append({
                    "image": str(path.relative_to(test_dir)),
                    "true_class": CATEGORIES[target],
                    "predicted_class": CATEGORIES[top1],
                    "confidence": float(probs[top1]),
                    "P_Bracelet": float(probs[0]), "P_Earring": float(probs[1]),
                    "P_Necklace": float(probs[2]), "P_Ring": float(probs[3]),
                    "top1": CATEGORIES[top1], "top2": CATEGORIES[top2],
                    "top1_probability": float(probs[top1]), "top2_probability": float(probs[top2]),
                    "margin": float(probs[top1] - probs[top2]),
                    "morph_circularity": float(morph[0]), "morph_aspect_ratio": float(morph[1]),
                    "morph_extent": float(morph[2]), "morph_solidity": float(morph[3]),
                })
            print(f"evaluated {min(start + len(group), len(samples))}/{len(samples)}", flush=True)

    targets = [CATEGORIES.index(row["true_class"]) for row in rows]
    predictions = [CATEGORIES.index(row["predicted_class"]) for row in rows]
    report = classification_report(targets, predictions, target_names=CATEGORIES, output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, predictions, labels=range(len(CATEGORIES)))
    metrics = {
        "protocol": "original RGB -> direct classifier preprocessing (pad_to_square, Resize(224,224), ImageNet normalization)",
        "morphology": "zeroed (visual-only ablation)" if zero_morph else "current morphology extraction",
        "seed": seed,
        "checkpoint": str(service.checkpoint_path),
        "test_images": len(rows),
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {name: report[name] for name in CATEGORIES},
        "confusion_matrix": cm.tolist(),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "seed": seed, "batch_size": batch_size, "checkpoint": str(service.checkpoint_path),
            "dataset": str(test_dir), "protocol": metrics["protocol"], "zero_morph": zero_morph, "training_performed": False,
        }, handle, indent=2)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "classification_report.txt").write_text(
        classification_report(targets, predictions, target_names=CATEGORIES, digits=4, zero_division=0)
        + "\nConfusion matrix (rows=true, columns=predicted)\n"
        + np.array2string(cm)
        + "\n",
        encoding="utf-8",
    )
    _save_confusion_matrix(cm, CATEGORIES, output_dir / "confusion_matrix.png")
    # The repository currently has no regression images. Keep the schema in
    # place so an empty set is explicit rather than silently omitted.
    with (output_dir / "regression_results.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=["image", "true_class", "predicted_class", "status"]).writeheader()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "experiments" / "A0_baseline")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-morph", action="store_true", help="Ablate morphology without modifying the checkpoint")
    arguments = parser.parse_args()
    print(json.dumps(evaluate(arguments.test_dir, arguments.output, arguments.batch_size, arguments.seed, arguments.zero_morph), indent=2))

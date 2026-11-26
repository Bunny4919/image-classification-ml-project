import os
import sys
import yaml
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.vision.similarity import load_threshold_config, CONFIG_PATH


def sweep_thresholds():
    print("\n" + "=" * 60)
    print("      THRESHOLD TUNING & UNKNOWN DETECTION EVALUATION      ")
    print("=" * 60)

    # Grouped test cases by category for per-category tuning
    category_test_cases = {
        "Ring": [
            {"true_status": "KNOWN_DESIGN", "cat_conf": 0.95, "sim_score": 0.88},
            {"true_status": "KNOWN_DESIGN", "cat_conf": 0.91, "sim_score": 0.82},
            {"true_status": "KNOWN_CATEGORY_UNKNOWN_DESIGN", "cat_conf": 0.89, "sim_score": 0.55},
            {"true_status": "UNSUPPORTED_OBJECT", "cat_conf": 0.35, "sim_score": 0.20},
        ],
        "Earring": [
            {"true_status": "KNOWN_DESIGN", "cat_conf": 0.94, "sim_score": 0.86},
            {"true_status": "KNOWN_CATEGORY_UNKNOWN_DESIGN", "cat_conf": 0.92, "sim_score": 0.61},
            {"true_status": "UNSUPPORTED_OBJECT", "cat_conf": 0.42, "sim_score": 0.30},
        ],
        "Necklace": [
            {"true_status": "KNOWN_DESIGN", "cat_conf": 0.88, "sim_score": 0.79},
            {"true_status": "KNOWN_CATEGORY_UNKNOWN_DESIGN", "cat_conf": 0.85, "sim_score": 0.48},
            {"true_status": "UNSUPPORTED_OBJECT", "cat_conf": 0.28, "sim_score": 0.15},
        ],
        "Bracelet": [
            {"true_status": "KNOWN_DESIGN", "cat_conf": 0.94, "sim_score": 0.86},
            {"true_status": "KNOWN_CATEGORY_UNKNOWN_DESIGN", "cat_conf": 0.89, "sim_score": 0.55},
            {"true_status": "UNSUPPORTED_OBJECT", "cat_conf": 0.35, "sim_score": 0.20},
        ],
    }

    cat_candidates = np.linspace(0.50, 0.85, 8)
    sim_candidates = np.linspace(0.50, 0.85, 8)

    best_cat_thresh = 0.70
    best_sim_thresholds = {}

    # 1. Find global best category confidence threshold
    # (Using all cases combined)
    all_cases = [tc for cats in category_test_cases.values() for tc in cats]
    best_acc = -1.0
    for cat_t in cat_candidates:
        correct = 0
        for tc in all_cases:
            pred_status = "UNSUPPORTED_OBJECT"
            if tc["cat_conf"] >= cat_t:
                # Use a temporary sim_t for the global sweep
                if tc["sim_score"] >= 0.75:
                    pred_status = "KNOWN_DESIGN"
                elif tc["true_status"] == "KNOWN_CATEGORY_UNKNOWN_DESIGN":
                    pred_status = "KNOWN_CATEGORY_UNKNOWN_DESIGN"
                elif tc["true_status"] == "KNOWN_DESIGN":
                    pred_status = "KNOWN_CATEGORY_UNKNOWN_DESIGN"

                # Simplification for global sweep:
                # If cat_conf is high, it's at least a KNOWN_CATEGORY.
                if tc["true_status"] == "UNSUPPORTED_OBJECT":
                    pass # Incorrect
                else:
                    correct += 1 # Correctly identified as supported

        acc = correct / len(all_cases)
        if acc > best_acc:
            best_acc = acc
            best_cat_thresh = round(float(cat_t), 2)

    print(f"Optimal Global Category Confidence Threshold: {best_cat_thresh}")

    # 2. Find per-category similarity thresholds
    for category, cases in category_test_cases.items():
        best_cat_acc = -1.0
        best_sim_t = 0.75

        for sim_t in sim_candidates:
            correct = 0
            for tc in cases:
                pred_status = "UNSUPPORTED_OBJECT"
                if tc["cat_conf"] >= best_cat_thresh:
                    if tc["sim_score"] >= sim_t:
                        pred_status = "KNOWN_DESIGN"
                    else:
                        pred_status = "KNOWN_CATEGORY_UNKNOWN_DESIGN"

                if pred_status == tc["true_status"]:
                    correct += 1

            acc = correct / len(cases)
            if acc > best_cat_acc:
                best_cat_acc = acc
                best_sim_t = round(float(sim_t), 2)

        best_sim_thresholds[category] = best_sim_t
        print(f"Category {category:<10} Optimal Sim Threshold: {best_sim_t} (Acc: {best_cat_acc*100:.1f}%)")

    # Update thresholds config
    config = load_threshold_config(CONFIG_PATH)
    config["category_confidence_threshold"] = best_cat_thresh
    config["similarity_thresholds"] = best_sim_thresholds
    # Keep legacy single threshold for backward compatibility
    config["similarity_threshold"] = float(np.mean(list(best_sim_thresholds.values())))

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    print(f"\nUpdated configuration saved to {CONFIG_PATH}\n")


if __name__ == "__main__":
    sweep_thresholds()

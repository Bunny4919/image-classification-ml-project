import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Add root directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.vision.preprocessing import classifier_transform, extract_morphological_features
from app.vision.classifier import ClassifierService, CATEGORIES

TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "classification", "test")


def evaluate_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test dataset directory {TEST_DATA_DIR} not found.")
        return

    test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=classifier_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    class_names = test_dataset.classes
    print(f"Evaluating classifier on {len(test_dataset)} test images across classes: {class_names}")

    service = ClassifierService(device=str(device))
    model = service.model

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # Extract morphological features for the batch
            # Since images are tensors, we need to convert them back to PIL or handle them as numpy
            # Simplest: the test_dataset is an ImageFolder, we can use its samples
            # But the loader already gave us the batch.
            # Let's iterate through the batch and extract features.

            morphs_list = []
            for img_tensor in images:
                # Convert tensor to PIL Image
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                morph_feats = extract_morphological_features(img_pil)
                morphs_list.append(morph_feats)

            morphs = torch.tensor(np.array(morphs_list), dtype=torch.float32).to(device)

            outputs = model(images, morphs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    total = len(all_targets)
    correct = np.sum(all_preds == all_targets)
    accuracy = correct / total if total > 0 else 0.0

    num_classes = len(class_names)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t, p] += 1

    print("\n" + "=" * 55)
    print("      CATEGORY CLASSIFIER EVALUATION REPORT      ")
    print("=" * 55)
    print(f"Overall Accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n")

    print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 55)

    f1_scores = []
    for i, cat_name in enumerate(class_names):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        print(f"{cat_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

    print("-" * 55)
    print(f"Macro F1 Score: {np.mean(f1_scores):.4f}\n")

    print("Confusion Matrix:")
    header = "          " + " ".join([f"{cat[:7]:>7}" for cat in class_names])
    print(header)
    for i, cat in enumerate(class_names):
        row_str = f"{cat[:8]:<8}  " + " ".join([f"{conf_matrix[i, j]:>7}" for j in range(num_classes)])
        print(row_str)
    print("=" * 55)

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist()
    }


if __name__ == "__main__":
    evaluate_classifier()

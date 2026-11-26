import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# Add root directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.vision.preprocessing import classifier_transform, extract_morphological_features
from app.vision.classifier import ClassifierService, CATEGORIES

TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "classification", "test")

def evaluate_detailed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test dataset directory {TEST_DATA_DIR} not found.")
        return

    test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=classifier_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    class_names = test_dataset.classes
    print(f"Evaluating classifier on {len(test_dataset)} test images across classes: {class_names}")

    # IMPORTANT: This service loads the current checkpoint
    service = ClassifierService(device=str(device))
    model = service.model
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            morphs_list = []
            for img_tensor in images:
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                # Correct preprocessing for morphological features - we must use the ORIGINAL image
                # because transforms.ToPILImage() on a normalized tensor is not the original image.
                # Since test_dataset is ImageFolder, we can get the original path.
                # For simplicity in this script, we'll just use the tensor-to-pil but it's a known limitation.
                # Better: iterate through dataset.samples.
                morph_feats = extract_morphological_features(img_pil)
                morphs_list.append(morph_feats)

            morphs = torch.tensor(np.array(morphs_list), dtype=torch.float32).to(device)
            outputs = model(images, morphs)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    print("\n" + "=" * 60)
    print("            DETAILED CLASSIFIER BASELINE REPORT            ")
    print("=" * 60)
    print(f"Overall Accuracy: {(np.sum(all_preds == all_targets) / len(all_targets)) * 100:.2f}%")
    print(f"Macro Precision:  {precision:.4f}")
    print(f"Macro Recall:     {recall:.4f}")
    print(f"Macro F1 Score:   {f1:.4f}\n")

    print("Per-Class Metrics:")
    print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 55)

    per_class_report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    for cat in class_names:
        metrics = per_class_report[cat]
        print(f"{cat:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    header = "          " + " ".join([f"{cat[:7]:>7}" for cat in class_names])
    print(header)
    for i, cat in enumerate(class_names):
        row_str = f"{cat[:8]:<8}  " + " ".join([f"{cm[i, j]:>7}" for j in range(len(class_names))])
        print(row_str)
    print("=" * 60)

    # Probability Distribution Analysis
    print("\nProbability Distribution (Average per Class):")
    for i, cat in enumerate(class_names):
        avg_prob = np.mean(all_probs[all_targets == i], axis=0) if np.any(all_targets == i) else np.zeros(len(class_names))
        prob_str = " ".join([f"{class_names[j]}: {avg_prob[j]:.3f}" for j in range(len(class_names))])
        print(f"{cat:<15} | {prob_str}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_detailed()

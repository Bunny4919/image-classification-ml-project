import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image

# Add root directory to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.vision.preprocessing import classifier_transform, train_classifier_transform, extract_morphological_features
from app.vision.classifier import JewelleryClassifier, CATEGORIES

INDEXES_DIR = os.path.join(BASE_DIR, "indexes")
CHECKPOINT_PATH = os.path.join(INDEXES_DIR, "classifier_resnet50.pth")
LEGACY_CHECKPOINT_PATH = os.path.join(INDEXES_DIR, "classifier_resnet18.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "classification")


class JewelleryDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = Image.open(path).convert("RGB")

        # Extract morphological features from original image
        morph_feats = extract_morphological_features(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(morph_feats, dtype=torch.float32), label

def train_model(epochs: int = 5, batch_size: int = 32, lr: float = 2e-4):
    os.makedirs(INDEXES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(f"Training high-accuracy ResNet50 classifier on device: {device}", flush=True)

    train_path = os.path.join(DATA_DIR, "train")
    val_path = os.path.join(DATA_DIR, "val")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training path {train_path} does not exist. Run data/prepare_dataset.py first.")

    train_dataset = JewelleryDataset(root_dir=train_path, transform=train_classifier_transform)
    val_dataset = JewelleryDataset(root_dir=val_path, transform=classifier_transform)

    # 1. Print class_to_idx mapping and verify classes
    print("\n--- DATASET INFORMATION ---", flush=True)
    print(f"Canonical Categories       : {CATEGORIES}", flush=True)
    print(f"ImageFolder classes loaded : {train_dataset.dataset.classes}", flush=True)
    print(f"ImageFolder class_to_idx   : {train_dataset.dataset.class_to_idx}", flush=True)
    assert train_dataset.dataset.classes == CATEGORIES, (
        f"Mismatch: expected classes {CATEGORIES}, got {train_dataset.dataset.classes}"
    )

    print(f"Training set size          : {len(train_dataset)} images", flush=True)
    print(f"Validation set size        : {len(val_dataset)} images", flush=True)

    # 2. Dynamic Class Imbalance Handling
    class_counts = torch.zeros(len(CATEGORIES), dtype=torch.float)
    for _, label in train_dataset.dataset.samples:
        class_counts[label] += 1

    total_samples = float(len(train_dataset.dataset.samples))
    num_classes = float(len(CATEGORIES))
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights.to(device)

    counts_dict = {c: int(cnt.item()) for c, cnt in zip(CATEGORIES, class_counts)}
    weights_dict = {c: round(float(w.item()), 4) for c, w in zip(CATEGORIES, class_weights)}
    print(f"Class Sample Counts        : {counts_dict}", flush=True)
    print(f"Dynamic Class Weights      : {weights_dict}", flush=True)
    print("---------------------------\n", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Model & Loss Definition
    model = JewelleryClassifier(num_classes=len(CATEGORIES), pretrained=True, backbone_name="resnet50")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    print(f"Starting training for {epochs} epochs (batch_size={batch_size}, lr={lr})...\n", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, morphs, labels) in enumerate(train_loader, 1):
            images, morphs, labels = images.to(device), morphs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, morphs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if step % 25 == 0 or step == len(train_loader):
                batch_acc = (preds == labels).float().mean().item()
                print(f"  Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f} Batch Acc: {batch_acc:.4f}", flush=True)

        scheduler.step()
        train_acc = correct / total if total > 0 else 0.0
        train_loss = running_loss / total if total > 0 else 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # For Macro-F1 calculation
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for images, morphs, labels in val_loader:
                images, morphs, labels = images.to(device), morphs.to(device), labels.to(device)
                outputs = model(images, morphs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(labels.numpy())

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss / val_total if val_total > 0 else 0.0

        # Compute Macro F1
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(all_val_targets, all_val_preds, average='macro')

        print(f"\n=== Epoch [{epoch}/{epochs}] Summary ===", flush=True)
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%", flush=True)
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc * 100:.2f}% | Val Macro-F1: {macro_f1 * 100:.2f}%", flush=True)

        if macro_f1 >= best_val_acc: # Using best_val_acc variable to track best Macro-F1
            best_val_acc = macro_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            # Maintain legacy copy so secondary consumers have access
            torch.save(model.state_dict(), LEGACY_CHECKPOINT_PATH)
            print(f"  --> Saved new BEST checkpoint (Val Macro-F1: {macro_f1 * 100:.2f}%) to {CHECKPOINT_PATH}", flush=True)
        print("=========================================\n", flush=True)

    print(f"Classifier training completed! Best Validation Accuracy: {best_val_acc * 100:.2f}%\n", flush=True)
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 Jewellery Classifier with Class Balancing")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

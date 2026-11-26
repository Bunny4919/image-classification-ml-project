import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    import pillow_avif
except ImportError:
    pass
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, f1_score

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.vision.preprocessing import (
    classifier_transform,
    train_classifier_transform,
    extract_morphological_features,
    pad_to_square,
)
from app.vision.classifier import JewelleryClassifier, CATEGORIES

INDEXES_DIR = BASE_DIR / "indexes"
B0_CHECKPOINT_PATH = INDEXES_DIR / "classifier_resnet50.pth"
B1_CHECKPOINT_PATH = INDEXES_DIR / "classifier_resnet50_b1.pth"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "B1_finetune"
DATA_DIR = BASE_DIR / "data" / "processed" / "classification"


def get_file_md5(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class B1JewelleryDataset(Dataset):
    """Combines original train dataset with verified hard negatives."""

    def __init__(self, samples: List[Tuple[Path, int]], transform=None, is_train: bool = True):
        self.samples = samples
        self.transform = transform
        self.is_train = is_train
        # Precompute morphology to speed up training epochs
        print(f"Precomputing morphology for {len(samples)} samples (is_train={is_train})...", flush=True)
        self.morph_features = []
        for path, _ in samples:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            self.morph_features.append(extract_morphological_features(rgb))
        print("Precomputation complete.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")

        morph = self.morph_features[idx]

        # Use pad_to_square before transforms to preserve aspect ratio
        rgb_padded = pad_to_square(rgb)
        if self.transform:
            image_tensor = self.transform(rgb_padded)
        else:
            image_tensor = classifier_transform(rgb_padded)

        return image_tensor, torch.tensor(morph, dtype=torch.float32), label


def verify_data_integrity():
    """Verify counts and zero-leakage constraints."""
    print("--- DATA INTEGRITY VERIFICATION ---", flush=True)
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"
    reg_dir = DATA_DIR / "regression"
    hn_dir = DATA_DIR / "hard_negatives"

    train_files = {get_file_md5(p): p for p in train_dir.glob("*/*") if p.is_file()}
    val_files = {get_file_md5(p): p for p in val_dir.glob("*/*") if p.is_file()}
    test_files = {get_file_md5(p): p for p in test_dir.glob("*/*") if p.is_file()}
    reg_files = {get_file_md5(p): p for p in reg_dir.glob("*/*") if p.is_file()}
    hn_files = {get_file_md5(p): p for p in hn_dir.glob("*/*") if p.is_file()}

    print(f"Original Train Count     : {len(train_files)} (Expected: 4995)")
    print(f"Validation Count         : {len(val_files)} (Expected: 548)")
    print(f"Test Count (Untouched)   : {len(test_files)} (Expected: 393)")
    print(f"Regression Count (Untouch): {len(reg_files)} (Expected: 10)")
    print(f"Hard Negatives Count     : {len(hn_files)} (Expected: 220)")

    assert len(train_files) == 4995, f"Train count mismatch: {len(train_files)}"
    assert len(val_files) == 548, f"Val count mismatch: {len(val_files)}"
    assert len(test_files) == 393, f"Test count mismatch: {len(test_files)}"
    assert len(reg_files) == 10, f"Regression count mismatch: {len(reg_files)}"
    assert len(hn_files) == 220, f"Hard negatives count mismatch: {len(hn_files)}"

    # Check for leakage
    assert len(set(hn_files.keys()) & set(test_files.keys())) == 0, "LEAKAGE: HN found in test set!"
    assert len(set(hn_files.keys()) & set(reg_files.keys())) == 0, "LEAKAGE: HN found in regression set!"
    assert len(set(hn_files.keys()) & set(val_files.keys())) == 0, "LEAKAGE: HN found in val set!"
    assert len(set(hn_files.keys()) & set(train_files.keys())) == 0, "DUPLICATE: HN found in train set!"

    print("Integrity verification passed: 0 leakage, 0 duplicates.\n", flush=True)


def train_b1(epochs: int = 6, batch_size: int = 32, lr_backbone: float = 1.5e-5, lr_head: float = 4e-5):
    verify_data_integrity()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    b0_hash_before = get_file_md5(B0_CHECKPOINT_PATH)
    print(f"B0 Checkpoint MD5 (Before Training): {b0_hash_before}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training B1 on device: {device}", flush=True)

    # 1. Collect training samples (Original train + Hard negatives)
    train_samples: List[Tuple[Path, int]] = []
    for cat_idx, cat in enumerate(CATEGORIES):
        cat_dir = DATA_DIR / "train" / cat
        for p in sorted(cat_dir.glob("*")):
            if p.is_file():
                train_samples.append((p, cat_idx))

    hn_counts = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        hn_cat_dir = DATA_DIR / "hard_negatives" / cat
        cat_hn = 0
        if hn_cat_dir.exists():
            for p in sorted(hn_cat_dir.glob("*")):
                if p.is_file():
                    train_samples.append((p, cat_idx))
                    cat_hn += 1
        hn_counts[cat] = cat_hn

    print(f"Hard negatives added: {hn_counts}", flush=True)
    print(f"Total B1 training samples: {len(train_samples)} (4995 original + 220 hard negatives)", flush=True)

    # 2. Collect validation samples
    val_samples: List[Tuple[Path, int]] = []
    for cat_idx, cat in enumerate(CATEGORIES):
        cat_dir = DATA_DIR / "val" / cat
        for p in sorted(cat_dir.glob("*")):
            if p.is_file():
                val_samples.append((p, cat_idx))
    print(f"Total validation samples: {len(val_samples)} (Untouched)", flush=True)

    # 3. Create Datasets and DataLoaders
    train_dataset = B1JewelleryDataset(train_samples, transform=train_classifier_transform, is_train=True)
    val_dataset = B1JewelleryDataset(val_samples, transform=classifier_transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Class counts and dynamic weights
    class_counts = torch.zeros(len(CATEGORIES), dtype=torch.float)
    for _, label in train_samples:
        class_counts[label] += 1

    total_samples = float(len(train_samples))
    num_classes = float(len(CATEGORIES))
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights.to(device)

    print("\n--- B1 CLASS DISTRIBUTION & LOSS WEIGHTS ---", flush=True)
    for cat, cnt, w in zip(CATEGORIES, class_counts, class_weights):
        print(f"  {cat:10s}: {int(cnt.item()):4d} images | Weight: {w.item():.4f}", flush=True)
    print("---------------------------------------------\n", flush=True)

    # 5. Initialize model from B0 checkpoint
    model = JewelleryClassifier(num_classes=len(CATEGORIES), pretrained=False, backbone_name="resnet50")
    b0_state = torch.load(B0_CHECKPOINT_PATH, map_location="cpu")
    # Remap keys if necessary
    remapped_sd = {}
    for k, v in b0_state.items():
        if k == "backbone.fc.weight":
            remapped_sd["backbone.fc.1.weight"] = v
        elif k == "backbone.fc.bias":
            remapped_sd["backbone.fc.1.bias"] = v
        else:
            remapped_sd[k] = v

    try:
        model.load_state_dict(remapped_sd)
    except Exception:
        model.load_state_dict(b0_state)
    print(f"Successfully initialized B1 from frozen B0 checkpoint: {B0_CHECKPOINT_PATH}", flush=True)
    model.to(device)

    # Safeguards against catastrophic forgetting:
    # Freeze conv1, bn1, layer1, layer2 to preserve early feature extractors
    for p in model.backbone.conv1.parameters():
        p.requires_grad = False
    for p in model.backbone.bn1.parameters():
        p.requires_grad = False
    for p in model.backbone.layer1.parameters():
        p.requires_grad = False
    for p in model.backbone.layer2.parameters():
        p.requires_grad = False

    # Fine-tune layer3, layer4 with low lr, fusion_head with slightly higher lr
    param_groups = [
        {"params": [p for p in model.backbone.layer3.parameters() if p.requires_grad], "lr": lr_backbone * 0.5},
        {"params": [p for p in model.backbone.layer4.parameters() if p.requires_grad], "lr": lr_backbone},
        {"params": [p for p in model.fusion_head.parameters() if p.requires_grad], "lr": lr_head},
    ]

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_macro_f1 = 0.0
    best_epoch = 0
    history = []

    print(f"\nStarting controlled B1 fine-tuning for {epochs} epochs...\n", flush=True)

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

            # Gradient clipping to prevent gradient shocks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if step % 40 == 0 or step == len(train_loader):
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
                all_val_targets.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss / val_total if val_total > 0 else 0.0
        macro_f1 = float(f1_score(all_val_targets, all_val_preds, average="macro"))

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": macro_f1,
        }
        history.append(epoch_stats)

        print(f"\n=== Epoch [{epoch}/{epochs}] Summary ===", flush=True)
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%", flush=True)
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc * 100:.2f}% | Val Macro-F1: {macro_f1 * 100:.2f}%", flush=True)

        if macro_f1 >= best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_epoch = epoch
            # Save B1 strictly to separate files
            torch.save(model.state_dict(), B1_CHECKPOINT_PATH)
            torch.save(model.state_dict(), EXPERIMENTS_DIR / "classifier_resnet50_b1.pth")
            print(f"  --> Saved new BEST B1 checkpoint (Val Macro-F1: {macro_f1 * 100:.2f}%) to {B1_CHECKPOINT_PATH}", flush=True)
        print("=========================================\n", flush=True)

    # Verify B0 checkpoint was NOT modified
    b0_hash_after = get_file_md5(B0_CHECKPOINT_PATH)
    print(f"B0 Checkpoint MD5 (After Training): {b0_hash_after}", flush=True)
    assert b0_hash_before == b0_hash_after, "FATAL ERROR: B0 checkpoint hash changed during B1 training!"
    print("IMMUTABILITY VERIFIED: B0 checkpoint is byte-for-byte identical.", flush=True)

    b1_hash = get_file_md5(B1_CHECKPOINT_PATH)
    print(f"B1 Checkpoint MD5: {b1_hash}", flush=True)

    # Save training record
    training_summary = {
        "b0_checkpoint": str(B0_CHECKPOINT_PATH),
        "b0_md5": b0_hash_before,
        "b1_checkpoint": str(B1_CHECKPOINT_PATH),
        "b1_md5": b1_hash,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_backbone": lr_backbone,
        "lr_head": lr_head,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
    }

    with open(EXPERIMENTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print(f"\nB1 Training completed! Best epoch: {best_epoch} with Val Macro-F1: {best_val_macro_f1 * 100:.2f}%", flush=True)
    return training_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controlled B1 Fine-Tuning with Hard Negatives")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1.5e-5)
    parser.add_argument("--lr-head", type=float, default=4e-5)
    args = parser.parse_args()

    train_b1(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
    )

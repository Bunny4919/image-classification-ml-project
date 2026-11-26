import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import pillow_avif
except ImportError:
    pass

from PIL import Image
from sklearn.metrics import f1_score

# ============================================================
# PROJECT PATHS
# ============================================================

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
B2_CHECKPOINT_PATH = INDEXES_DIR / "classifier_resnet50_b2.pth"

EXPERIMENTS_DIR = BASE_DIR / "experiments" / "B2_training"

DATA_DIR = BASE_DIR / "data" / "processed" / "classification"

ORIGINAL_TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
REGRESSION_DIR = DATA_DIR / "regression"

B1_HARD_NEGATIVE_DIR = DATA_DIR / "hard_negatives"
B2_HARD_NEGATIVE_DIR = DATA_DIR / "hard_negatives_b2"


# ============================================================
# UTILITY
# ============================================================

def get_file_md5(filepath: Path) -> str:
    """Return MD5 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ============================================================
# DATASET
# ============================================================

class B2JewelleryDataset(Dataset):
    """
    Dataset used for B2.

    Combines:
        - original training images
        - B1 hard negatives
        - B2 hard negatives

    Validation uses the untouched validation set.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform=None,
        is_train: bool = True,
    ):
        self.samples = samples
        self.transform = transform
        self.is_train = is_train

        print(
            f"Precomputing morphology for {len(samples)} samples "
            f"(is_train={is_train})...",
            flush=True,
        )

        self.morph_features = []

        for path, _ in samples:
            with Image.open(path) as img:
                rgb = img.convert("RGB")

            self.morph_features.append(
                extract_morphological_features(rgb)
            )

        print("Precomputation complete.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            rgb = img.convert("RGB")

        morph = self.morph_features[idx]

        # Preserve aspect ratio before classifier transformation.
        rgb_padded = pad_to_square(rgb)

        if self.transform:
            image_tensor = self.transform(rgb_padded)
        else:
            image_tensor = classifier_transform(rgb_padded)

        return (
            image_tensor,
            torch.tensor(morph, dtype=torch.float32),
            label,
        )


# ============================================================
# DATA INTEGRITY
# ============================================================

def verify_data_integrity():
    """
    Verify B2 dataset counts and zero-leakage constraints.

    Expected:
        Original train       = 4995
        Validation           = 548
        Test                 = 393
        Regression           = 10
        B1 hard negatives    = 220
        B2 hard negatives    = 160
    """

    print("=" * 70)
    print("B2 DATA INTEGRITY VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Collect MD5 hashes
    # --------------------------------------------------------

    train_files = {
        get_file_md5(p): p
        for p in ORIGINAL_TRAIN_DIR.glob("*/*")
        if p.is_file()
    }

    val_files = {
        get_file_md5(p): p
        for p in VAL_DIR.glob("*/*")
        if p.is_file()
    }

    test_files = {
        get_file_md5(p): p
        for p in TEST_DIR.glob("*/*")
        if p.is_file()
    }

    regression_files = {
        get_file_md5(p): p
        for p in REGRESSION_DIR.glob("*/*")
        if p.is_file()
    }

    b1_hn_files = {
        get_file_md5(p): p
        for p in B1_HARD_NEGATIVE_DIR.glob("*/*")
        if p.is_file()
    }

    b2_hn_files = {
        get_file_md5(p): p
        for p in B2_HARD_NEGATIVE_DIR.glob("*/*")
        if p.is_file()
    }

    # --------------------------------------------------------
    # Print counts
    # --------------------------------------------------------

    print(
        f"Original Train Count : {len(train_files)} "
        f"(Expected: 4995)"
    )

    print(
        f"Validation Count     : {len(val_files)} "
        f"(Expected: 548)"
    )

    print(
        f"Test Count           : {len(test_files)} "
        f"(Expected: 393)"
    )

    print(
        f"Regression Count     : {len(regression_files)} "
        f"(Expected: 10)"
    )

    print(
        f"B1 Hard Negatives    : {len(b1_hn_files)} "
        f"(Expected: 220)"
    )

    print(
        f"B2 Hard Negatives    : {len(b2_hn_files)} "
        f"(Expected: 160)"
    )

    # --------------------------------------------------------
    # Count assertions
    # --------------------------------------------------------

    assert len(train_files) == 4995, (
        f"Train count mismatch: {len(train_files)}"
    )

    assert len(val_files) == 548, (
        f"Validation count mismatch: {len(val_files)}"
    )

    assert len(test_files) == 393, (
        f"Test count mismatch: {len(test_files)}"
    )

    assert len(regression_files) == 10, (
        f"Regression count mismatch: {len(regression_files)}"
    )

    assert len(b1_hn_files) == 220, (
        f"B1 hard-negative count mismatch: {len(b1_hn_files)}"
    )

    assert len(b2_hn_files) == 160, (
        f"B2 hard-negative count mismatch: {len(b2_hn_files)}"
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    print("\nChecking B1 hard-negative leakage...")

    assert not (
        set(b1_hn_files) & set(train_files)
    ), "LEAKAGE: B1 hard negatives overlap training set!"

    assert not (
        set(b1_hn_files) & set(val_files)
    ), "LEAKAGE: B1 hard negatives overlap validation set!"

    assert not (
        set(b1_hn_files) & set(test_files)
    ), "LEAKAGE: B1 hard negatives overlap test set!"

    assert not (
        set(b1_hn_files) & set(regression_files)
    ), "LEAKAGE: B1 hard negatives overlap regression set!"

    print("B1 hard-negative leakage: PASS")

    print("\nChecking B2 hard-negative leakage...")

    assert not (
        set(b2_hn_files) & set(train_files)
    ), "LEAKAGE: B2 hard negatives overlap training set!"

    assert not (
        set(b2_hn_files) & set(val_files)
    ), "LEAKAGE: B2 hard negatives overlap validation set!"

    assert not (
        set(b2_hn_files) & set(test_files)
    ), "LEAKAGE: B2 hard negatives overlap test set!"

    assert not (
        set(b2_hn_files) & set(regression_files)
    ), "LEAKAGE: B2 hard negatives overlap regression set!"

    assert not (
        set(b2_hn_files) & set(b1_hn_files)
    ), "DUPLICATE: B2 hard negatives overlap B1 hard negatives!"

    print("B2 hard-negative leakage: PASS")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA INTEGRITY: PASS")
    print("No train/val/test/regression leakage detected.")
    print("No B1/B2 hard-negative overlap detected.")
    print("=" * 70 + "\n")


# ============================================================
# MODEL CHECKPOINT LOADING
# ============================================================

def load_checkpoint_into_model(model, checkpoint_path: Path):
    """
    Load a checkpoint while handling the known backbone.fc
    naming difference.
    """

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    remapped_state = {}

    for key, value in state.items():

        if key == "backbone.fc.weight":
            remapped_state["backbone.fc.1.weight"] = value

        elif key == "backbone.fc.bias":
            remapped_state["backbone.fc.1.bias"] = value

        else:
            remapped_state[key] = value

    try:
        model.load_state_dict(remapped_state)
    except Exception:
        model.load_state_dict(state)

    return model


# ============================================================
# B2 TRAINING
# ============================================================

def train_b2(
    epochs: int = 6,
    batch_size: int = 32,
    lr_backbone: float = 1.5e-5,
    lr_head: float = 4e-5,
):
    """
    Controlled B2 fine-tuning.

    Initialization:
        B1 checkpoint

    Training:
        4995 original
        + 220 B1 hard negatives
        + 160 B2 hard negatives
        = 5375 images

    Validation:
        untouched 548 images
    """

    # --------------------------------------------------------
    # Verify data before touching model
    # --------------------------------------------------------

    verify_data_integrity()

    # --------------------------------------------------------
    # Verify checkpoints exist
    # --------------------------------------------------------

    assert B0_CHECKPOINT_PATH.exists(), (
        f"Missing B0 checkpoint: {B0_CHECKPOINT_PATH}"
    )

    assert B1_CHECKPOINT_PATH.exists(), (
        f"Missing B1 checkpoint: {B1_CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INDEXES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Record B0/B1 hashes BEFORE training
    # --------------------------------------------------------

    b0_hash_before = get_file_md5(
        B0_CHECKPOINT_PATH
    )

    b1_hash_before = get_file_md5(
        B1_CHECKPOINT_PATH
    )

    print(
        f"B0 Checkpoint MD5 Before Training: "
        f"{b0_hash_before}",
        flush=True,
    )

    print(
        f"B1 Checkpoint MD5 Before Training: "
        f"{b1_hash_before}",
        flush=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nTraining B2 on device: {device}",
        flush=True,
    )

    if device.type == "cuda":
        print(
            f"CUDA Device: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    # ========================================================
    # 1. COLLECT ORIGINAL TRAINING DATA
    # ========================================================

    train_samples: List[Tuple[Path, int]] = []

    for cat_idx, cat in enumerate(CATEGORIES):

        cat_dir = ORIGINAL_TRAIN_DIR / cat

        for path in sorted(cat_dir.glob("*")):

            if path.is_file():

                train_samples.append(
                    (path, cat_idx)
                )

    print(
        f"\nOriginal training samples: "
        f"{len(train_samples)}",
        flush=True,
    )

    # ========================================================
    # 2. ADD B1 HARD NEGATIVES
    # ========================================================

    b1_hn_counts = {}

    for cat_idx, cat in enumerate(CATEGORIES):

        hn_cat_dir = (
            B1_HARD_NEGATIVE_DIR / cat
        )

        count = 0

        if hn_cat_dir.exists():

            for path in sorted(
                hn_cat_dir.glob("*")
            ):

                if path.is_file():

                    train_samples.append(
                        (path, cat_idx)
                    )

                    count += 1

        b1_hn_counts[cat] = count

    print(
        f"B1 hard negatives added: "
        f"{b1_hn_counts}",
        flush=True,
    )

    # ========================================================
    # 3. ADD B2 HARD NEGATIVES
    # ========================================================

    b2_hn_counts = {}

    for cat_idx, cat in enumerate(CATEGORIES):

        b2_cat_dir = (
            B2_HARD_NEGATIVE_DIR / cat
        )

        count = 0

        if b2_cat_dir.exists():

            for path in sorted(
                b2_cat_dir.glob("*")
            ):

                if path.is_file():

                    train_samples.append(
                        (path, cat_idx)
                    )

                    count += 1

        b2_hn_counts[cat] = count

    print(
        f"B2 hard negatives added: "
        f"{b2_hn_counts}",
        flush=True,
    )

    # --------------------------------------------------------
    # Verify final training count
    # --------------------------------------------------------

    expected_training_count = 5375

    assert len(train_samples) == expected_training_count, (
        f"Training sample mismatch: "
        f"{len(train_samples)} != "
        f"{expected_training_count}"
    )

    print(
        f"\nTotal B2 training samples: "
        f"{len(train_samples)}",
        flush=True,
    )

    print(
        "  Original train       : 4995",
        flush=True,
    )

    print(
        "  B1 hard negatives    : 220",
        flush=True,
    )

    print(
        "  B2 hard negatives    : 160",
        flush=True,
    )

    print(
        "  TOTAL                : 5375",
        flush=True,
    )

    # ========================================================
    # 4. VALIDATION DATA
    # ========================================================

    val_samples: List[Tuple[Path, int]] = []

    for cat_idx, cat in enumerate(CATEGORIES):

        cat_dir = VAL_DIR / cat

        for path in sorted(cat_dir.glob("*")):

            if path.is_file():

                val_samples.append(
                    (path, cat_idx)
                )

    assert len(val_samples) == 548, (
        f"Validation sample mismatch: "
        f"{len(val_samples)} != 548"
    )

    print(
        f"\nValidation samples: "
        f"{len(val_samples)} (UNTouched)",
        flush=True,
    )

    # ========================================================
    # 5. DATASETS
    # ========================================================

    print("\nCreating training dataset...", flush=True)

    train_dataset = B2JewelleryDataset(
        train_samples,
        transform=train_classifier_transform,
        is_train=True,
    )

    print("\nCreating validation dataset...", flush=True)

    val_dataset = B2JewelleryDataset(
        val_samples,
        transform=classifier_transform,
        is_train=False,
    )

    # ========================================================
    # 6. DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"\nTraining batches: {len(train_loader)}",
        flush=True,
    )

    print(
        f"Validation batches: {len(val_loader)}",
        flush=True,
    )

    # ========================================================
    # 7. DYNAMIC CLASS WEIGHTS
    # ========================================================

    class_counts = torch.zeros(
        len(CATEGORIES),
        dtype=torch.float,
    )

    for _, label in train_samples:

        class_counts[label] += 1

    total_samples = float(
        len(train_samples)
    )

    num_classes = float(
        len(CATEGORIES)
    )

    class_weights = (
        total_samples
        / (num_classes * class_counts)
    )

    class_weights = class_weights.to(
        device
    )

    print(
        "\n--- B2 CLASS DISTRIBUTION & LOSS WEIGHTS ---",
        flush=True,
    )

    for cat, count, weight in zip(
        CATEGORIES,
        class_counts,
        class_weights,
    ):

        print(
            f"  {cat:10s}: "
            f"{int(count.item()):4d} images | "
            f"Weight: {weight.item():.4f}",
            flush=True,
        )

    print(
        "---------------------------------------------\n",
        flush=True,
    )

    # ========================================================
    # 8. INITIALIZE FROM B1
    # ========================================================

    model = JewelleryClassifier(
        num_classes=len(CATEGORIES),
        pretrained=False,
        backbone_name="resnet50",
    )

    print(
        f"Loading B1 checkpoint:\n"
        f"{B1_CHECKPOINT_PATH}",
        flush=True,
    )

    model = load_checkpoint_into_model(
        model,
        B1_CHECKPOINT_PATH,
    )

    print(
        "Successfully initialized B2 "
        "from B1 checkpoint.",
        flush=True,
    )

    model.to(device)

    # ========================================================
    # 9. FREEZE EARLY BACKBONE
    # ========================================================

    print(
        "\nApplying catastrophic-forgetting safeguards...",
        flush=True,
    )

    for p in model.backbone.conv1.parameters():
        p.requires_grad = False

    for p in model.backbone.bn1.parameters():
        p.requires_grad = False

    for p in model.backbone.layer1.parameters():
        p.requires_grad = False

    for p in model.backbone.layer2.parameters():
        p.requires_grad = False

    print(
        "Frozen: conv1, bn1, layer1, layer2",
        flush=True,
    )

    print(
        "Trainable: layer3, layer4, fusion_head",
        flush=True,
    )

    # ========================================================
    # 10. OPTIMIZER
    # ========================================================

    param_groups = [
        {
            "params": [
                p
                for p in model.backbone.layer3.parameters()
                if p.requires_grad
            ],
            "lr": lr_backbone * 0.5,
        },
        {
            "params": [
                p
                for p in model.backbone.layer4.parameters()
                if p.requires_grad
            ],
            "lr": lr_backbone,
        },
        {
            "params": [
                p
                for p in model.fusion_head.parameters()
                if p.requires_grad
            ],
            "lr": lr_head,
        },
    ]

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    # ========================================================
    # 11. TRAINING STATE
    # ========================================================

    best_val_macro_f1 = 0.0
    best_epoch = 0

    history = []

    print(
        "\n" + "=" * 70,
        flush=True,
    )

    print(
        f"STARTING B2 FINE-TUNING "
        f"FOR {epochs} EPOCHS",
        flush=True,
    )

    print(
        "=" * 70 + "\n",
        flush=True,
    )

    # ========================================================
    # 12. TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for step, (
            images,
            morphs,
            labels,
        ) in enumerate(
            train_loader,
            1,
        ):

            images = images.to(device)
            morphs = morphs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                images,
                morphs,
            )

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            _, preds = torch.max(
                outputs,
                1,
            )

            total += labels.size(0)

            correct += (
                preds == labels
            ).sum().item()

            if (
                step % 40 == 0
                or step == len(train_loader)
            ):

                batch_acc = (
                    preds == labels
                ).float().mean().item()

                print(
                    f"  Epoch [{epoch}/{epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Batch Acc: {batch_acc:.4f}",
                    flush=True,
                )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        train_acc = (
            correct / total
            if total > 0
            else 0.0
        )

        train_loss = (
            running_loss / total
            if total > 0
            else 0.0
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():

            for (
                images,
                morphs,
                labels,
            ) in val_loader:

                images = images.to(device)
                morphs = morphs.to(device)
                labels = labels.to(device)

                outputs = model(
                    images,
                    morphs,
                )

                loss = criterion(
                    outputs,
                    labels,
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                _, preds = torch.max(
                    outputs,
                    1,
                )

                val_total += labels.size(0)

                val_correct += (
                    preds == labels
                ).sum().item()

                all_val_preds.extend(
                    preds.cpu().numpy()
                )

                all_val_targets.extend(
                    labels.cpu().numpy()
                )

        val_acc = (
            val_correct / val_total
            if val_total > 0
            else 0.0
        )

        val_loss = (
            val_loss / val_total
            if val_total > 0
            else 0.0
        )

        macro_f1 = float(
            f1_score(
                all_val_targets,
                all_val_preds,
                average="macro",
            )
        )

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": macro_f1,
            "learning_rates": [
                group["lr"]
                for group in optimizer.param_groups
            ],
        }

        history.append(
            epoch_stats
        )

        print(
            f"\n=== Epoch [{epoch}/{epochs}] Summary ===",
            flush=True,
        )

        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}%",
            flush=True,
        )

        print(
            f"  Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Macro-F1: {macro_f1 * 100:.2f}%",
            flush=True,
        )

        # ----------------------------------------------------
        # SAVE BEST B2 CHECKPOINT
        # ----------------------------------------------------

        if macro_f1 >= best_val_macro_f1:

            best_val_macro_f1 = macro_f1
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                B2_CHECKPOINT_PATH,
            )

            torch.save(
                model.state_dict(),
                EXPERIMENTS_DIR
                / "classifier_resnet50_b2.pth",
            )

            print(
                f"  --> Saved new BEST B2 checkpoint "
                f"(Val Macro-F1: "
                f"{macro_f1 * 100:.2f}%)",
                flush=True,
            )

            print(
                f"      {B2_CHECKPOINT_PATH}",
                flush=True,
            )

        print(
            "=========================================\n",
            flush=True,
        )

    # ========================================================
    # 13. VERIFY B0 AND B1 IMMUTABILITY
    # ========================================================

    b0_hash_after = get_file_md5(
        B0_CHECKPOINT_PATH
    )

    b1_hash_after = get_file_md5(
        B1_CHECKPOINT_PATH
    )

    print(
        "\n" + "=" * 70,
        flush=True,
    )

    print(
        "CHECKPOINT IMMUTABILITY VERIFICATION",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    print(
        f"B0 Before: {b0_hash_before}",
        flush=True,
    )

    print(
        f"B0 After : {b0_hash_after}",
        flush=True,
    )

    print(
        f"B1 Before: {b1_hash_before}",
        flush=True,
    )

    print(
        f"B1 After : {b1_hash_after}",
        flush=True,
    )

    assert (
        b0_hash_before
        == b0_hash_after
    ), (
        "FATAL ERROR: "
        "B0 checkpoint changed during B2 training!"
    )

    assert (
        b1_hash_before
        == b1_hash_after
    ), (
        "FATAL ERROR: "
        "B1 checkpoint changed during B2 training!"
    )

    print(
        "\nIMMUTABILITY VERIFIED: "
        "B0 and B1 are byte-for-byte identical.",
        flush=True,
    )

    # ========================================================
    # 14. VERIFY B2 CHECKPOINT
    # ========================================================

    assert B2_CHECKPOINT_PATH.exists(), (
        "FATAL ERROR: B2 checkpoint was not created!"
    )

    b2_hash = get_file_md5(
        B2_CHECKPOINT_PATH
    )

    print(
        f"\nB2 Checkpoint MD5: {b2_hash}",
        flush=True,
    )

    # ========================================================
    # 15. SAVE TRAINING SUMMARY
    # ========================================================

    training_summary = {
        "experiment": "B2_finetune",

        "initialization": {
            "checkpoint": str(
                B1_CHECKPOINT_PATH
            ),
            "md5": b1_hash_before,
        },

        "b0_checkpoint": str(
            B0_CHECKPOINT_PATH
        ),

        "b0_md5_before": b0_hash_before,

        "b0_md5_after": b0_hash_after,

        "b1_checkpoint": str(
            B1_CHECKPOINT_PATH
        ),

        "b1_md5_before": b1_hash_before,

        "b1_md5_after": b1_hash_after,

        "b2_checkpoint": str(
            B2_CHECKPOINT_PATH
        ),

        "b2_md5": b2_hash,

        "epochs": epochs,

        "batch_size": batch_size,

        "lr_backbone": lr_backbone,

        "lr_head": lr_head,

        "original_train_samples": 4995,

        "b1_hard_negative_samples": 220,

        "b2_hard_negative_samples": 160,

        "total_train_samples": len(
            train_samples
        ),

        "validation_samples": len(
            val_samples
        ),

        "b1_hard_negative_counts": b1_hn_counts,

        "b2_hard_negative_counts": b2_hn_counts,

        "best_epoch": best_epoch,

        "best_val_macro_f1": best_val_macro_f1,

        "device": str(device),

        "history": history,
    }

    summary_path = (
        EXPERIMENTS_DIR
        / "training_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            training_summary,
            f,
            indent=2,
        )

    # ========================================================
    # 16. FINAL STATUS
    # ========================================================

    print(
        "\n" + "=" * 70,
        flush=True,
    )

    print(
        "B2 TRAINING COMPLETED",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    print(
        f"Best Epoch       : {best_epoch}",
        flush=True,
    )

    print(
        f"Best Val Macro-F1: "
        f"{best_val_macro_f1 * 100:.2f}%",
        flush=True,
    )

    print(
        f"B2 Checkpoint    : "
        f"{B2_CHECKPOINT_PATH}",
        flush=True,
    )

    print(
        f"B2 MD5           : {b2_hash}",
        flush=True,
    )

    print(
        f"Training Summary  : {summary_path}",
        flush=True,
    )

    print(
        "\nB0 unchanged: PASS",
        flush=True,
    )

    print(
        "B1 unchanged: PASS",
        flush=True,
    )

    print(
        "B2 checkpoint: CREATED",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    return training_summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Controlled B2 Fine-Tuning "
            "with B1 + B2 Hard Negatives"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1.5e-5,
    )

    parser.add_argument(
        "--lr-head",
        type=float,
        default=4e-5,
    )

    args = parser.parse_args()

    train_b2(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
    )
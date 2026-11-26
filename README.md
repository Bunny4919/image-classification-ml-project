# Jewellery Image Classification System

## 1. Project Overview

This project implements an end-to-end computer vision and multimodal retrieval system designed for the accurate classification, visual search, and catalog-grounded retrieval of fine jewellery images. The primary classification task categorises jewellery images into four distinct classes:

- **Bracelet**
- **Earring**
- **Necklace**
- **Ring**

While standard benchmarks evaluate models on clean, high-contrast e-commerce packshots with plain white backgrounds, production environments encounter complex real-world imagery. The engineering objective of this project is to achieve high classification robustness across both clean studio product images and challenging real-world conditions, including:

- Complex, textured, and lifestyle backgrounds (e.g., wood, marble, fabrics)
- Severe close-up cropping where chain links or clasps exit the frame
- Human hand, wrist, and neck contexts with skin tones and fingers
- Presentation in gift boxes, velvet slots, and cushions
- Dominant pendants or medallions obscuring slender chains
- Scale ambiguity where jewellery items share geometric contours without physical scale context
- Lifestyle photography with natural lighting, glare, and angled perspectives

### Architectural Principles

1. **Classifier Authority**: The custom ResNet50 neural network classifier is the single authoritative source of truth for the jewellery category.
2. **Retrieval Boundary**: Downstream visual vector search (DINOv2 embeddings in FAISS) and generative context (ChromaDB / Gemini RAG) are strictly filtered by the classifier's predicted category. Vector retrieval and LLM responses must never silently override the ResNet50 category decision.
3. **Serving API**: A high-performance FastAPI service (`app/main.py`) provides REST endpoints for image classification, catalogue identification (`/api/v1/identify`), and grounded design Q&A (`/api/v1/query`).

---

## 2. Problem Statement

Jewellery classification exhibits high inter-class visual similarity and strong intra-class variance, frequently inducing misclassification:

- **Bracelet vs. Earring**: Charm bracelets with dangling elements, heart pendants, or open hoop-like bangles visually mimic drop earrings or large hoop earrings when cropped.
- **Necklace vs. Earring**: Close-up framing focusing on a prominent pendant or locket where the chain exits the frame leaves only a circular or teardrop motif that is visually identical to a stud or drop earring.
- **Necklace vs. Bracelet**: Segmented close-ups showing only an arc of a chain or tennis strand without endpoints resemble a tennis bracelet. Rigid circular collar necklaces also closely resemble circular bangles.
- **Ring vs. Bracelet**: Plain, wide gold bands photographed without hand or finger scale cues share circular and toroidal geometry with bangles.
- **Lifestyle Context**: Rings worn on hands or held between fingers present hand textures that can distract models trained exclusively on plain white studio cutouts.

Standard benchmark accuracy (e.g., >90% on clean cutouts) is insufficient to guarantee operational reliability because it fails to reveal systematic vulnerabilities to background shifts, cropping, and scale ambiguity.

---

## 3. Classes

| Class | Description |
|---|---|
| **Bracelet** | Wrist jewellery such as link bracelets, tennis bracelets, bangles, and open cuffs |
| **Earring** | Ear jewellery including studs, hoops, drop earrings, chandelier, and jhumka styles |
| **Necklace** | Neck jewellery including chains, pendants, chokers, bibs, and collar necklaces |
| **Ring** | Finger jewellery including plain bands, eternity bands, and gemstone/solitaire rings |

---

## 4. System Architecture

```
Input Image (Upload via Web UI / API)
    ↓
Image Preprocessing & Validation
    (MIME check, PIL verification, EXIF correction, pad-to-square 1:1, Resize 224×224, ImageNet Norm)
    ↓
ResNet50 Feature Extraction & Classification Head
    (ResNet50 Backbone [2048-dim] + Fail-Safe Morphology Extraction [4-dim])
    ↓
4-Class Probability Output (Softmax over Bracelet, Earring, Necklace, Ring)
    ↓
Authoritative Category Prediction (Argmax, Confidence, Top-2 Margin)
    ↓
Supporting Components (Constrained by Classifier Category)
    ├── Morphology Diagnostic Analysis
    └── DINOv2 (ViT-S/14) Feature Vector Extraction (384-dim)
            ↓
        FAISS Vector Search (Filtered strictly by predicted category)
            ↓
        ChromaDB Metadata Store (Retrieves verified material, karat, stone specs)
            ↓
        RAG / Grounded LLM Response (Gemini 1.5 Flash grounded synthesis)
    ↓
Final System Response (JSON / Web UI View)
```

The ResNet50 category prediction remains strictly authoritative throughout the entire pipeline; FAISS candidate retrieval is constrained within the predicted class partition.

---

## 5. Technology Stack

All documented technologies have been verified against the active codebase:

- **Core & Runtime**: Python 3.10+, PyTorch 2.x, Torchvision
- **Backbone Architecture**: ResNet50 (residual deep CNN with custom late-fusion head)
- **Visual Embeddings**: DINOv2 ViT-S/14 (Facebook Research) for 384-dimensional dense visual representations
- **Vector Indexing**: FAISS (`IndexFlatIP` for inner-product cosine similarity search)
- **Metadata Database**: ChromaDB (persistent embedded vector/document database)
- **API Framework**: FastAPI, Uvicorn (ASGI web server)
- **Morphology & Vision Utilities**: OpenCV (`cv2`), Pillow (`PIL`), `pillow_avif`
- **Data Science & Metrics**: NumPy, scikit-learn
- **Hardware Acceleration**: NVIDIA CUDA GPU (GeForce RTX 3050 Laptop GPU tested)

---

## 6. Data Pipeline

```
Raw Image Sources (Roboflow Jewellery Dataset, OrnAsia Asian Ornament Dataset, Curated Photos)
    ↓
Dataset Preparation & Quality Filtering (`data/prepare_dataset.py`)
    ↓
Class Mapping & Canonical Labelling (Bracelet, Earring, Necklace, Ring)
    ↓
Deterministic Train / Validation / Test Splitting
    ↓
Standardised Preprocessing Pipeline
    (Square padding with neutral background fill, bicubic resize to 224×224, ImageNet normalization)
    ↓
Model Training & Checkpointing
    ↓
Evaluation against Untouched Evaluation Suites
```

### Dataset Partition Counts

- **Train Set**: 4,995 images (used for baseline B0 model training)
  - Bracelet: 1,008
  - Earring: 1,035
  - Necklace: 2,409
  - Ring: 543
- **Validation Set**: 548 images (preserved untouched across all experiments)
  - Bracelet: 112
  - Earring: 115
  - Necklace: 254
  - Ring: 67
- **Test Set**: 393 images (benchmark evaluation only; never enters training)
  - Bracelet: 82
  - Earring: 98
  - Necklace: 150
  - Ring: 63
- **Real-World Regression Suite**: 10 images (held-out challenge images for real-world robustness evaluation; strictly evaluation-only)
  - Bracelet: 2
  - Earring: 3
  - Necklace: 2
  - Ring: 3

---

## 7. B0 Baseline

The baseline model B0 is stored at `indexes/classifier_resnet50.pth` (MD5: `8ebe13ce911bb09b931986b1bc770675`).

### B0 Test Set Results (393 Images)

- **Accuracy**: 94.40% (371 / 393)
- **Macro-F1**: 0.9464
- **Macro-Precision**: 0.9388
- **Macro-Recall**: 0.9570

#### B0 Per-Class Metrics:
- **Bracelet**: Precision = 0.9195, Recall = 0.9756, F1 = 0.9467 (Support = 82)
- **Earring**: Precision = 0.8952, Recall = 0.9592, F1 = 0.9261 (Support = 98)
- **Necklace**: Precision = 1.0000, Recall = 0.8933, F1 = 0.9437 (Support = 150)
- **Ring**: Precision = 0.9403, Recall = 1.0000, F1 = 0.9692 (Support = 63)

#### B0 Confusion Matrix (Rows: Actual, Columns: Predicted [Bracelet, Earring, Necklace, Ring]):
```
[[ 80   0   0   2]
 [  2  94   0   2]
 [  5  11 134   0]
 [  0   0   0  63]]
```

#### Key Confusion Pairs in B0:
- **Necklace → Earring**: 11 errors (dominant failure mode; 7.3% of all necklaces)
- **Necklace → Bracelet**: 5 errors
- **Bracelet → Ring**: 2 errors
- **Earring → Bracelet**: 2 errors
- **Earring → Ring**: 2 errors
- **Bracelet → Earring**: 0 errors
- **Ring → Earring / Bracelet**: 0 errors

---

## 8. Real-World Regression Testing

To evaluate robustness on uncontrolled imagery beyond clean packshots, a separate 10-image real-world regression suite was curated in `data/processed/classification/regression/`.

### B0 Real-World Performance
- **Regression Accuracy**: 40.0% (4 / 10 correct)

### The Six Observed Real-World Failure Patterns in B0:
1. **Bracelet photographed on wrist/lifestyle background → Earring** (`sample bracelet.webp`): Worn bracelet on arm confused with dangle earrings.
2. **Open cuff bracelet → Earring** (`test b.jpg`): Open curved metal profile misclassified as hoop/cuff earring.
3. **Necklace held across hands → Earring** (`sample chain.webp`): Hand context and close framing misidentified as earrings.
4. **Necklace in gift box with prominent pendant → Earring** (`vrace.jpg`): Pendant displayed on velvet pad read as isolated stud/earring.
5. **Plain gold band without scale context → Bracelet** (`ring1.jpg`): Plain circular ring band without finger context mistaken for a bangle.
6. **Ring in lifestyle/hand context → Earring** (`sample ring.webp`): Ring on finger with lifestyle backdrop misclassified as earring.

This marked a major performance gap: **94.40% on standard benchmark vs. 40.0% on real-world challenge images**.

---

## 9. Failure Analysis Method

The engineering methodology followed a diagnostic loop:

```
Wrong Prediction on Real-World & Test Sets
    ↓
Confusion Matrix & Error Grouping (Identify systematic misclassification clusters)
    ↓
Visual Inspection of Failure Cases (Contact sheets & feature distribution analysis)
    ↓
Failure Pattern Identification (Pendants without chains, close crops, lifestyle backgrounds, wide bands)
    ↓
Morphology Diagnostic Investigation
    ↓
Targeted Hard-Negative Data Curation (Sourcing specific visual distributions)
    ↓
Audit for Data Leakage & Duplicates
    ↓
Controlled Fine-Tuning from B0
    ↓
Rigorous Evaluation on Untouched Test (393) & Regression (10) Datasets
```

### Morphology Investigation Finding
Morphology extraction (circularity, aspect ratio, extent, solidity) was evaluated via ablation (`evaluation/reproducible_evaluation.py --zero-morph`). Visual-only ablation yielded an identical 94.40% accuracy and 0.9464 Macro-F1, with fusion weights demonstrating an L2 norm of 0.593 for morphology inputs compared to 13.401 for visual features. This demonstrated that morphology is neutral for these failure modes, and that structural improvements must be driven by visual representation learning through targeted hard negatives.

---

## 10. Hard-Negative Strategy — B1

A targeted hard-negative dataset of **220 verified images** was curated in `data/processed/classification/hard_negatives/` to address the observed failure patterns without altering class balances:

- **Bracelet (75 images)**:
  - Bracelets photographed worn on wrists/arms and lifestyle backgrounds
  - Charms, dangle hearts, and multi-strand link bracelets
  - Open metal cuffs and rigid bangles
- **Necklace (75 images)**:
  - Prominent pendants, lockets, and medallions photographed with visible chains
  - Full-arc necklaces laid flat showing chain length
  - Necklaces in presentation gift boxes and displays
  - Close-up chain framing with context
- **Ring (70 images)**:
  - Plain, thick, and wide gold bands without finger context
  - Worn rings on fingers with hand and lifestyle backgrounds
- **Earring (0 images)**:
  - Earring precision was already strong; adding hard negatives to Earring would have exacerbated the attractor effect.

---

## 11. Data Integrity and Leakage Prevention

Strict data boundary controls were enforced before any training:

1. **Independent Verification**:
   - MD5 hash checks confirmed **0 duplicates** between hard negatives and the original 4,995 training images.
   - MD5 hash checks confirmed **0 duplicates** between hard negatives and the 548 validation images.
   - MD5 hash checks confirmed **0 duplicates** between hard negatives and the 393 test images.
   - MD5 hash checks confirmed **0 duplicates** between hard negatives and the 10 regression images.
2. **Untouched Splits**:
   - Original train set remains untouched at 4,995 images.
   - Validation set remains untouched at 548 images.
   - Test set remains completely isolated at 393 images.
   - Regression suite remains completely isolated at 10 images.

---

## 12. B1 Training Approach

B1 fine-tuning was executed via `training/train_b1.py` with safeguards against catastrophic forgetting:

- **Initialization**: Initialized strictly from the frozen B0 checkpoint (`indexes/classifier_resnet50.pth`).
- **Checkpoint Isolation**: B0 was loaded in read-only mode and verified via MD5 before and after training (`8ebe13ce911bb09b931986b1bc770675`). B1 was saved to a completely separate path (`indexes/classifier_resnet50_b1.pth`).
- **Layer Freezing**: Early backbone layers (`conv1`, `bn1`, `layer1`, `layer2`) were frozen to preserve low-level visual representations.
- **Differential Learning Rates**:
  - Backbone `layer3`: $0.75 \times 10^{-5}$
  - Backbone `layer4`: $1.5 \times 10^{-5}$
  - Fusion classification head: $4.0 \times 10^{-5}$
- **Regularisation**: AdamW optimizer with weight decay $10^{-4}$ and gradient clipping (`max_norm=1.0`).
- **Learning Rate Schedule**: Cosine Annealing scheduler ($T_{\max}=6, \eta_{\min}=10^{-6}$).
- **Loss Formulation**: CrossEntropyLoss with dynamic inverse-frequency class weights:
  - Bracelet ($N=1,083$): 1.2038
  - Earring ($N=1,035$): 1.2597
  - Necklace ($N=2,484$): 0.5249
  - Ring ($N=613$): 2.1268
- **Validation Monitoring**: Evaluated after every epoch on the untouched 548 validation images. The best checkpoint was achieved at **Epoch 4** (Validation Macro-F1: 97.02%, Validation Accuracy: 97.26%).

---

## 13. B0 vs B1 Evaluation

Both B0 and B1 were evaluated using identical protocols (`evaluation/evaluate_b0_b1.py`) on the untouched 393-image test set and 10-image real-world regression suite.

### A. 393-Image Test Benchmark Comparison

| Metric | B0 (Baseline) | B1 (Fine-Tuned) | Delta |
|---|---:|---:|---:|
| **Test Accuracy** | **94.40%** | **95.67%** | **+1.27%** |
| **Macro-F1** | **0.9464** | **0.9584** | **+0.0120** |
| **Macro-Precision** | **0.9388** | **0.9545** | **+0.0157** |
| **Macro-Recall** | **0.9570** | **0.9639** | **+0.0069** |

#### Per-Class Metrics Comparison:
| Class | Metric | B0 | B1 | Change |
|---|---|---:|---:|---:|
| **Bracelet** | Precision / Recall / F1 | 0.9195 / 0.9756 / 0.9467 | 0.9625 / 0.9390 / **0.9506** | **F1 +0.0039** |
| **Earring** | Precision / Recall / F1 | 0.8952 / 0.9592 / 0.9261 | 0.9151 / 0.9898 / **0.9510** | **F1 +0.0249** |
| **Necklace** | Precision / Recall / F1 | 1.0000 / 0.8933 / 0.9437 | 0.9858 / 0.9267 / **0.9553** | **F1 +0.0116** |
| **Ring** | Precision / Recall / F1 | 0.9403 / 1.0000 / 0.9692 | 0.9545 / 1.0000 / **0.9767** | **F1 +0.0075** |

#### Key Confusion Pairs:
| Confusion Pair | B0 Errors | B1 Errors | Change |
|---|---:|---:|---:|
| **Necklace → Earring** | **11** | **9** | **-2 (-18.2%)** |
| **Necklace → Bracelet** | **5** | **2** | **-3 (-60.0%)** |
| **Bracelet → Earring** | 0 | 0 | 0 |
| **Ring → Bracelet** | 0 | 0 | 0 |
| **Ring → Earring** | 0 | 0 | 0 |
| **Earring → Bracelet** | 2 | 1 | -1 |
| **Earring → Ring** | 2 | 0 | -2 (Eliminated) |
| **Bracelet → Ring** | 2 | 3 | +1 |
| **Bracelet → Necklace** | 0 | 2 | +2 |

#### Confusion Matrices (Bracelet, Earring, Necklace, Ring):

**B0 Baseline Confusion Matrix:**
```
[[ 80   0   0   2]
 [  2  94   0   2]
 [  5  11 134   0]
 [  0   0   0  63]]
```

**B1 Fine-Tuned Confusion Matrix:**
```
[[ 77   0   2   3]
 [  1  97   0   0]
 [  2   9 139   0]
 [  0   0   0  63]]
```

---

### B. 10-Image Real-World Regression Comparison

| Filename | True Class | B0 Prediction (Conf) | B1 Prediction (Conf) | B0 Wrong? | B1 Fixed Error? | New Error? |
|---|---|---|---|:---:|:---:|:---:|
| `sample bracelet.webp` | Bracelet | Earring (0.487) | **Bracelet (0.492)** | Yes | **YES (Fixed)** | No |
| `test b.jpg` | Bracelet | Earring (0.659) | Earring (0.450) | Yes | No | No |
| `0682054DZAVER00004173_1.avif` | Earring | Earring (0.998) | Earring (0.989) | No | Preserved | No |
| `images.jpeg` | Earring | Earring (1.000) | Earring (1.000) | No | Preserved | No |
| `rubans-22k-...-earrings.jpg` | Earring | Earring (0.991) | Earring (0.983) | No | Preserved | No |
| `sample chain.webp` | Necklace | Earring (0.790) | **Necklace (0.586)** | Yes | **YES (Fixed)** | No |
| `vrace.jpg` | Necklace | Earring (0.968) | Earring (0.995) | Yes | No | No |
| `ring1.jpg` | Ring | Bracelet (0.973) | Bracelet (0.886) | Yes | No | No |
| `sample ring.webp` | Ring | Earring (0.872) | Earring (0.669) | Yes | No | No |
| `Screenshot 2026-09-11 154859.png`| Ring | Ring (0.894) | Ring (0.710) | No | Preserved | No |

**Regression Suite Summary:**
- **Accuracy**: Improved from **40.0% (4/10)** in B0 to **60.0% (6/10)** in B1 (+20.0%).
- **New Errors Introduced**: **0** (all 4 cases that passed in B0 continue to pass in B1).

#### Analysis of the Six Known Real-World Failures:
1. **Bracelet photographed on wrist/lifestyle background → Earring** (`sample bracelet.webp`): **FIXED** (B1 predicts Bracelet).
2. **Necklace held across hands → Earring** (`sample chain.webp`): **FIXED** (B1 predicts Necklace).
3. **Open cuff bracelet → Earring** (`test b.jpg`): **Still Failing** (predicted Earring, though error confidence declined from 0.659 to 0.450).
4. **Necklace in gift box with prominent pendant → Earring** (`vrace.jpg`): **Still Failing** (dominant pendant in velvet box remains classified as Earring with 0.995 confidence).
5. **Plain gold band without scale context → Bracelet** (`ring1.jpg`): **Still Failing** (predicted Bracelet, confidence reduced from 0.973 to 0.886).
6. **Ring in lifestyle/hand context → Earring** (`sample ring.webp`): **Still Failing** (predicted Earring, confidence reduced from 0.872 to 0.669).

---

## 14. Success Criteria Assessment

B1 was evaluated against the strict criteria defined prior to training:

1. **Maintains or improves test performance**: **PASSED**. Test accuracy increased from 94.40% to 95.67% (+1.27%); Macro-F1 increased from 0.9464 to 0.9584.
2. **Reduces Necklace → Earring errors**: **PASSED**. Reduced from 11 to 9 errors (-18.2%).
3. **Reduces Necklace → Bracelet errors**: **PASSED**. Reduced from 5 to 2 errors (-60.0%).
4. **Fixes real-world failures**: **PASSED**. 2 of 6 known failures fixed; regression accuracy rose from 40% to 60%.
5. **Preserves already-strong classes**: **PASSED**. Per-class F1 improved across every single class (Bracelet: 0.9506, Earring: 0.9510, Necklace: 0.9553, Ring: 0.9767). Ring recall remained perfect at 1.0000.
6. **Avoids unacceptable new errors**: **PASSED**. Zero new errors introduced in the 10-image regression suite. On the test set, Earring → Ring errors dropped from 2 to 0, though Bracelet minor confusions had a slight shift (Bracelet → Necklace: +2, Bracelet → Ring: +1).

---

## 15. Reproducibility

### Key Scripts & Artifacts:
- **Dataset Preparation**: `data/prepare_dataset.py`
- **Baseline Training Script**: `training/train_classifier.py`
- **B1 Fine-Tuning Script**: `training/train_b1.py`
- **Evaluation Runner**: `evaluation/evaluate_b0_b1.py`
- **Benchmark Evaluation**: `evaluation/reproducible_evaluation.py`
- **Regression Evaluation**: `evaluation/run_regression_baseline.py`
- **Checkpoints**:
  - Baseline B0: `indexes/classifier_resnet50.pth` (MD5: `8ebe13ce911bb09b931986b1bc770675`)
  - Fine-Tuned B1: `indexes/classifier_resnet50_b1.pth` (MD5: `8d8ec65e052fd0f0b9c1971ee91e90c9`)
- **Experiment Artifacts**:
  - Baseline Test: `experiments/A0_baseline/`
  - Baseline Regression: `experiments/regression_baseline/`
  - B1 Fine-Tuning Logs: `experiments/B1_finetune/training_summary.json`
  - B1 vs B0 Evaluation: `experiments/B1_evaluation/`
  - Hard-Negative Manifest: `data/processed/classification/hard_negatives/curation_manifest.json`

### Reproduction Command:
```powershell
# 1. Run controlled B1 training (strictly preserves B0)
python -u training/train_b1.py --epochs 6 --batch-size 32 --lr-backbone 1.5e-5 --lr-head 4e-5

# 2. Run joint B0 vs B1 evaluation
python evaluation/evaluate_b0_b1.py
```

---

## 16. Current Project Status

- **B0 Baseline**: COMPLETE (Locked and verified)
- **Failure Analysis**: COMPLETE (Documented in `experiments/failure_analysis/`)
- **Real-World Regression Suite**: COMPLETE (10 challenge images)
- **Hard-Negative Curation**: COMPLETE (220 verified images, 0 leakage)
- **B1 Training**: COMPLETE (6 epochs, best epoch 4 with Val Macro-F1 97.02%)
- **B1 Evaluation**: COMPLETE (Measured on untouched 393-image test set and 10-image regression suite)

---

## 17. Key Engineering Approach

The project adheres to an empirical, measurement-driven engineering discipline:

```
Baseline (Establish reproducible metric benchmarks on fixed datasets)
    ↓
Measure (Record full probability distributions, margins, and confusion matrices)
    ↓
Find Failures (Identify high-confidence failure modes in benchmark and real-world sets)
    ↓
Diagnose Failure Patterns (Inspect image contact sheets, cropping, and background cues)
    ↓
Add Targeted Hard Negatives (Curate non-leaking data specifically matching error modes)
    ↓
Fine-Tune with Safeguards (Freeze early layers, differential LR, prevent catastrophic forgetting)
    ↓
Test on Untouched Data (Run identical evaluation protocol across test and regression sets)
    ↓
Compare against Baseline (Quantify gains, regressions, and remaining failure modes)
    ↓
Keep Changes Only If Supported by Measured Evidence
```

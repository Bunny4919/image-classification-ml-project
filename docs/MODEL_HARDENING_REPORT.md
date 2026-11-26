# Jewellery Classifier Hardening Report

## 1. Original Architecture

The authoritative category classifier is `app/vision/classifier.py`:
ResNet50 without its final FC layer produces 2,048 visual features, which are
concatenated with four morphology values and passed through
`Linear(2052, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 4)`. Class order is
`Bracelet`, `Earring`, `Necklace`, `Ring`.

The FastAPI route in `app/api/routes_identification.py` centrally crops an
upload, evaluates seven spatial crops, and averages their probabilities. It
then sends the selected crop to DINOv2 (`app/vision/embedding.py`), FAISS
(`app/retrieval/search.py`), and the RAG layer (`app/rag/*`). FAISS is filtered
by the classifier category; it does not overwrite that category.

## 2. Baseline Results

The baseline was actually run against the current `classifier_resnet50.pth` on
393 test images. The protocol uses source RGB images, original-image morphology,
and the direct classifier preprocessing path (square pad, 224x224 resize,
ImageNet normalisation). It deliberately does not apply API multi-view crops.

| Metric | Result |
|---|---:|
| Accuracy | 94.40% |
| Macro precision | 0.9388 |
| Macro recall | 0.9570 |
| Macro F1 | 0.9464 |
| Bracelet F1 | 0.9467 |
| Earring F1 | 0.9261 |
| Necklace F1 | 0.9437 |
| Ring F1 / recall | 0.9692 / 1.0000 |

Artifacts: `experiments/A0_baseline/` and `experiments/A2_mask_quality/`.

Confusion matrix (rows actual, columns predicted; Bracelet, Earring, Necklace,
Ring):

```
[[80,  0,   0, 2],
 [ 2, 94,   0, 2],
 [ 5, 11, 134, 0],
 [ 0,  0,   0, 63]]
```

Thus the present checkpoint's largest measured errors are Necklace -> Earring
(11) and Necklace -> Bracelet (5). Ring -> Earring is zero in this benchmark.
The older 88.30% baseline in the engineering brief was not reproducible with
this checkpoint and protocol.

## 3. Failure Analysis

### Bracelet vs Necklace

There are five Necklace -> Bracelet benchmark errors and no Bracelet ->
Necklace errors. This benchmark alone cannot diagnose real-world bracelet
failures.

### Ring vs Earring

There are no Ring -> Earring errors in the 393-image benchmark. Claims about
this failure mode require the missing real-world regression images.

### Morphology Failure

The former implementation used one inverse Otsu mask, assumed its largest
contour was jewellery, and supplied no mask-quality checks. It could produce a
valid OpenCV `uint8` mask representing background. Its direct concatenation
also gave no way for the old checkpoint to know morphology was unreliable.

### Background Sensitivity

The API has a separate centre-crop plus seven-crop weighted inference policy,
whereas training and direct benchmark evaluation have no such policy. This is a
material evaluation/inference mismatch. Background sensitivity has not been
quantified because no real-world regression images are present.

## 4. Changes Implemented

`app/vision/preprocessing.py` now provides fail-safe morphology extraction:

- candidate masks from inverse/normal Otsu, adaptive thresholding, and Canny;
- binary 0/255 `np.uint8` coercion before every OpenCV mask operation;
- opening and closing cleanup, connected-component measures, foreground ratio,
  largest-component share, and border-contact validation;
- neutral four-value morphology when no mask is plausible; and
- `extract_morphology_debug()` for masks, provenance, and quality diagnostics.

The function remains four-dimensional to preserve the frozen checkpoint's
input contract. A learned validity feature needs a newly trained fusion model;
it must not be appended to the current checkpoint at inference time.

`evaluation/reproducible_evaluation.py` produces deterministic metrics,
probabilities, top-2 classes, margins, CSV, report, and confusion-matrix PNG.
It also supports the visual-only (`--zero-morph`) ablation.

`tests/test_morphology.py` covers binary mask dtype, blank-image fallback, and
RGBA safety.

## 5. Experiment Matrix

| Experiment | Training | Accuracy | Macro F1 | Result |
|---|---:|---:|---:|---|
| A0 baseline | no | 94.40% | 0.9464 | direct classifier baseline |
| A1 morphology candidates | no | 94.40% | 0.9464 | no benchmark change |
| A2 stricter mask quality | no | 94.40% | 0.9464 | no benchmark change |
| visual-only ablation | no | 94.40% | 0.9464 | identical predictions |

The first fusion-layer morphology columns have L2 norm 0.593 versus 13.401 for
visual columns. Together with the exact visual-only result, this is evidence
that morphology is negligible for this checkpoint on this benchmark. A
morphology-only classifier cannot be meaningfully evaluated from this fused
checkpoint because it has no independent morphology head.

## 6. Confusion Matrix Analysis

The benchmark supports a Necklace-focused analysis, not a Ring-focused one:
11/150 necklaces are predicted Earring (7.3%) and 5/150 Bracelet (3.3%).
No test-derived tuning has been performed.

## 7. Hard-Negative Results

No hard-negative data was added or retrained. The repository has an old
`scratch_audit_results.json` containing four Bracelet -> Earring entries (two
train, two validation) among its ranked items; it is not a permanent six-image
regression set.

## 8. Regression Results

`data/processed/classification/regression/` exists but contains zero images.
Therefore the requested six-case results, real-world distribution comparison,
and regression pass rate cannot be measured. Empty-schema CSVs are emitted by
the evaluator rather than fabricating outcomes.

## 9. Background Robustness

Not run: there is no supplied real-world regression image set to transform and
no validated target behaviour for such transformations.

## 10. Confidence Analysis

Every direct benchmark prediction now records full probabilities, top-1,
top-2, confidence, and margin in `predictions.csv`. Threshold selection has
not been performed because it must use validation data and an explicit
ambiguous/acceptable target definition.

## 11. API Validation

The code validates MIME type, size, PIL decodability, EXIF orientation, and RGB
conversion. The new morphology path returns neutral features instead of
propagating OpenCV segmentation exceptions. The full pytest invocation is
currently blocked by a malformed root filesystem entry named
`CUserskonat.claudeplans `, which raises `FileNotFoundError` during pytest
collection. The morphology tests were invoked directly and pass.

## 12. Final Model

The final checkpoint remains the original `indexes/classifier_resnet50.pth`.
No retraining, architecture replacement, class remapping, or FAISS/RAG category
override was performed.

## 13. Remaining Limitations

The stricter segmentation policy is heuristic and its validity is not yet a
learned model input. Its real-world effect cannot be asserted without the six
images. The direct benchmark and API multi-view behaviour are not comparable.

## 14. Reproduction Instructions

Run:

```powershell
python -m evaluation.reproducible_evaluation --output experiments/A0_baseline --batch-size 4
python -m evaluation.reproducible_evaluation --output experiments/A2_visual_only --batch-size 4 --zero-morph
```

Add the original six labelled images under
`data/processed/classification/regression/<Class>/` before any retraining or
hard-negative model-selection claim.

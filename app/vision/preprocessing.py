import io
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import numpy as np
import cv2

# Supported image MIME types

# Supported image MIME types
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB default limit

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Classifier evaluation / inference transform
# Resize(256) preserves aspect ratio; CenterCrop(224) avoids stretching.
# Stretching a long necklace/chain into a square makes it look like a bracelet.
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),             # Direct resize to preserve whole object after pad_to_square
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Classifier training transform with data augmentation
train_classifier_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1)
    ),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(180),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# DINOv2 embedding transform (224x224)
embedding_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def fix_exif_orientation(image: Image.Image) -> Image.Image:
    """Corrects image orientation based on EXIF metadata."""
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


def load_and_verify_image(file_bytes: bytes, filename: str = "") -> Image.Image:
    """
    Validates and loads raw image bytes.
    Enforces maximum size limit and Pillow verification.
    Returns RGB PIL Image with EXIF orientation fixed.
    """
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.")

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    # Integrity verification
    try:
        bio = io.BytesIO(file_bytes)
        img = Image.open(bio)
        img.verify()
    except Exception as e:
        raise ValueError(f"Uploaded file is not a valid decodable image: {e}")

    # Re-open after verify() as per Pillow documentation
    bio = io.BytesIO(file_bytes)
    img = Image.open(bio)
    img = fix_exif_orientation(img)
    return img.convert("RGB")


def pad_to_square(image: Image.Image, fill_color=(255, 255, 255)) -> Image.Image:
    """Pads an image to a 1:1 square ratio with background fill before resizing to prevent aspect-ratio distortion."""
    w, h = image.size
    if w == h:
        return image
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), fill_color)
    new_img.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    return new_img


def preprocess_for_classification(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image into a normalized tensor batch (1, 3, 224, 224) for classifier."""
    squared = pad_to_square(image)
    tensor = classifier_transform(squared)
    return tensor.unsqueeze(0)


def preprocess_for_embedding(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image into a normalized tensor batch (1, 3, 224, 224) for vision encoder."""
    squared = pad_to_square(image)
    tensor = embedding_transform(squared)
    return tensor.unsqueeze(0)


NEUTRAL_MORPHOLOGY = np.zeros(4, dtype=np.float32)


def _binary_uint8(mask: np.ndarray) -> np.ndarray:
    """Return a two-dimensional 0/255 uint8 mask safe for OpenCV."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Morphology mask must be two-dimensional")
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    if mask.size == 0:
        return mask
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = _binary_uint8(mask)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _mask_quality(mask: np.ndarray) -> Dict[str, Any]:
    """Measure whether one connected region is plausibly an isolated object."""
    mask = _binary_uint8(mask)
    height, width = mask.shape
    image_area = max(height * width, 1)
    foreground_ratio = float(np.count_nonzero(mask) / image_area)
    components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if components > 1 else np.array([], dtype=np.int32)
    component_count = int(len(areas))
    largest_area = int(areas.max()) if component_count else 0
    largest_ratio = float(largest_area / image_area)
    touches_border = bool(
        np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
    )
    largest_share = largest_ratio / foreground_ratio if foreground_ratio else 0.0
    # A large border-connected region is normally background leakage, not an
    # isolated jewellery object. The bounds intentionally reject doubtful
    # masks; visual features are safer than misleading contour statistics.
    valid = bool(
        0.002 <= foreground_ratio <= 0.45
        and largest_ratio >= 0.003
        and largest_share >= 0.35
        and not (touches_border and foreground_ratio > 0.15)
    )
    return {
        "foreground_ratio": foreground_ratio,
        "component_count": component_count,
        "largest_component_ratio": largest_ratio,
        "largest_component_share": largest_share,
        "touches_border": touches_border,
        "valid": valid,
    }


def extract_morphology_debug(image: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract four checkpoint-compatible features plus mask diagnostics.

    The existing checkpoint was trained with exactly four morphology inputs, so
    invalid masks return a neutral all-zero vector rather than silently changing
    the model interface. A later retrained fusion model may consume ``valid``
    explicitly as a fifth input.
    """
    debug: Dict[str, Any] = {"valid": False, "reason": "unknown"}
    try:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu_inverse = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, otsu_normal = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4
        )
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        candidates = {
            "otsu_inverse": _clean_mask(otsu_inverse),
            "otsu_normal": _clean_mask(otsu_normal),
            "adaptive_inverse": _clean_mask(adaptive),
            "canny": _clean_mask(edges),
        }
        scored = []
        for name, mask in candidates.items():
            quality = _mask_quality(mask)
            # Prefer a compact single candidate, then penalise near-full masks.
            compactness = quality["largest_component_ratio"] / max(quality["foreground_ratio"], 1e-9)
            score = compactness - abs(quality["foreground_ratio"] - 0.15)
            if quality["touches_border"]:
                score -= 0.10
            if not quality["valid"]:
                score -= 10.0
            scored.append((score, name, mask, quality))
        _, method, mask, quality = max(scored, key=lambda item: item[0])
        debug.update(quality)
        debug["method"] = method
        debug["mask"] = mask
        if not quality["valid"]:
            debug["reason"] = "no_plausible_foreground_region"
            return NEUTRAL_MORPHOLOGY.copy(), debug
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            debug["reason"] = "no_contour"
            return NEUTRAL_MORPHOLOGY.copy(), debug
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, width, height = cv2.boundingRect(contour)
        if area <= 0 or perimeter <= 0 or width <= 0 or height <= 0:
            debug["reason"] = "degenerate_contour"
            return NEUTRAL_MORPHOLOGY.copy(), debug
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        features = np.asarray([
            (4.0 * np.pi * area) / (perimeter * perimeter),
            max(width, height) / min(width, height),
            area / float(width * height),
            area / hull_area if hull_area > 0 else 0.0,
        ], dtype=np.float32)
        if not np.isfinite(features).all():
            debug["reason"] = "non_finite_features"
            return NEUTRAL_MORPHOLOGY.copy(), debug
        debug.update({"valid": True, "reason": "ok", "bbox": [int(x), int(y), int(width), int(height)]})
        return features, debug
    except Exception as error:
        debug["reason"] = f"exception:{type(error).__name__}"
        return NEUTRAL_MORPHOLOGY.copy(), debug


def extract_morphological_features(image: Image.Image) -> np.ndarray:
    """Return safe, checkpoint-compatible morphology features for inference."""
    features, _ = extract_morphology_debug(image)
    return features

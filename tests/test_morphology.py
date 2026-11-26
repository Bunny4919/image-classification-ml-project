import numpy as np
from PIL import Image

from app.vision.preprocessing import _binary_uint8, extract_morphology_debug, extract_morphological_features


def test_binary_mask_is_uint8_and_binary():
    mask = _binary_uint8(np.array([[0.0, 0.5], [1.0, 255.0]], dtype=np.float64))
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    assert set(np.unique(mask)).issubset({0, 255})


def test_blank_image_has_finite_checkpoint_compatible_features():
    features, debug = extract_morphology_debug(Image.new("RGB", (48, 48), "white"))
    assert features.shape == (4,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert debug["mask"].dtype == np.uint8


def test_morphology_never_raises_for_rgba_image():
    image = Image.new("RGBA", (32, 64), (0, 0, 0, 0))
    features = extract_morphological_features(image)
    assert features.shape == (4,)
    assert np.isfinite(features).all()

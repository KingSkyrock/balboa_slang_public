import numpy as np
from PIL import Image
from pathlib import Path

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    )

def save_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = linear_to_srgb(image[..., :3])
    alpha = np.clip(image[..., 3:4], 0.0, 1.0)

    rgba = np.concatenate([rgb, alpha], axis=-1)
    rgba8 = np.round(np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)

    Image.fromarray(rgba8, 'RGBA').save(path)
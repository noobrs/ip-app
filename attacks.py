import numpy as np
from PIL import Image

# =========================
# JPEG Attack
# =========================
def jpeg_attack(in_path, out_path, quality=80):
    Image.open(in_path).convert("RGB").save(out_path, quality=quality, subsampling=0, optimize=False)

# =========================
# Crop Attack
# =========================
def crop_attack(in_arr, area_ratio=0.01, num_patches=2, seed=42):
    """Tiny cutouts while keeping size; seed only for reproducibility of the ATTACK (not watermark)."""
    H, W, C = in_arr.shape
    rng = np.random.default_rng(seed)

    def rect_mask(x0, y0, w, h):
        m = np.zeros((H, W), dtype=np.uint8); m[y0:y0+h, x0:x0+w] = 255; return m

    for _ in range(max(1, int(num_patches))):
        w = max(1, int(round(np.sqrt(area_ratio) * W)))
        h = max(1, int(round(np.sqrt(area_ratio) * H)))
        x0 = int(rng.integers(0, max(1, W - w))); y0 = int(rng.integers(0, max(1, H - h)))
        mask = rect_mask(x0, y0, w, h)
        in_arr[mask == 255] = 0

    return Image.fromarray(in_arr)

# =========================
# Rotation Attack
# =========================
def rotation_attack(pil_img, angle=5.0, fill_color=(0, 0, 0)):
    """
    Rotate image by given angle in degrees.
    Positive angle = clockwise rotation.
    Fill color is used for areas outside the original image boundaries.
    """
    return pil_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill_color)

# =========================
# Additive Noise Attack
# =========================
def gaussian_noise_attack(in_arr, std=10, seed=None):
    """Add Gaussian noise to the image array."""
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, std, in_arr.shape)
    noisy = np.clip(in_arr.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def uniform_noise_attack(in_arr, intensity=20, seed=None):
    """Add uniform noise to the image array."""
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.uniform(-intensity, intensity, in_arr.shape)
    noisy = np.clip(in_arr.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)
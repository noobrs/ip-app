import math
import cv2 as cv
import numpy as np
from PIL import Image
import pywt
from scipy.fftpack import dct, idct
from parameters import WATERMARK_SIZE, DWT_LEVELS, WAVELET_NAME, WAVELET_MODE

# =========================
# PNSR
# =========================
def psnr(img_a_u8, img_b_u8):
    a = img_a_u8.astype(np.float64)
    b = img_b_u8.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))

# =========================
# BER
# =========================
def bit_error_rate(true_bits_u8, pred_bits_u8, return_counts=False):
    true_bits_u8 = np.asarray(true_bits_u8).astype(np.uint8).ravel()
    pred_bits_u8 = np.asarray(pred_bits_u8).astype(np.uint8).ravel()
    assert true_bits_u8.size == pred_bits_u8.size, "Mismatch in bit lengths."
    
    errors = np.count_nonzero(true_bits_u8 ^ pred_bits_u8)
    total = true_bits_u8.size
    ber = errors / total
    
    if return_counts:
        accuracy = 1.0 - ber
        return ber, {"errors": errors, "total": total, "accuracy": accuracy}
    return ber

# =========================
# NCC
# =========================
def normalized_cross_correlation(true_bits_u8, pred_bits_u8):
    """Compute Normalized Cross-Correlation between two binary watermarks."""
    true_bits_u8 = np.asarray(true_bits_u8).astype(np.float64).ravel()
    pred_bits_u8 = np.asarray(pred_bits_u8).astype(np.float64).ravel()
    assert true_bits_u8.size == pred_bits_u8.size, "Mismatch in bit lengths."
    
    # Center the data (subtract mean)
    true_centered = true_bits_u8 - np.mean(true_bits_u8)
    pred_centered = pred_bits_u8 - np.mean(pred_bits_u8)
    
    # Compute NCC
    numerator = np.sum(true_centered * pred_centered)
    denominator = np.sqrt(np.sum(true_centered**2) * np.sum(pred_centered**2))
    
    # Avoid division by zero
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

# =========================
# Image Conversion
# =========================
def to_ycbcr_arrays(pil_rgb):
    """Return Y (float64), Cb (uint8), Cr (uint8)."""
    ycbcr = pil_rgb.convert("YCbCr")
    Y  = np.array(ycbcr.getchannel(0), dtype=np.float64)
    Cb = np.array(ycbcr.getchannel(1), dtype=np.uint8)
    Cr = np.array(ycbcr.getchannel(2), dtype=np.uint8)
    return Y, Cb, Cr

def from_ycbcr_arrays(Y_float, Cb_u8, Cr_u8):
    """Merge Y, Cb, Cr back to RGB (uint8)."""
    Y_u8 = np.rint(np.clip(Y_float, 0, 255)).astype(np.uint8)
    ycbcr = Image.merge("YCbCr", (Image.fromarray(Y_u8),
                                  Image.fromarray(Cb_u8),
                                  Image.fromarray(Cr_u8)))
    return ycbcr.convert("RGB")

def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)

def np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

def to_pil_luma(y):
    """Ensure we have a single-channel PIL Image in 'L' mode."""
    if isinstance(y, Image.Image):
        return y.convert('L')
    arr = np.asarray(y)
    if arr.ndim != 2:
        raise ValueError("y_luma must be 2D (single-channel).")
    # Normalize dtype for PIL
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
        else:
            arr = np.round(arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')

# =========================
# Convert Watermark into Binary
# =========================
def prepare_watermark_bits(pil, size=WATERMARK_SIZE):
    """Load any image → grayscale → Otsu → resize to size×size, return 0/1 bits."""
    
    # composite onto white to handle transparency
    # _bg  = Image.new("RGBA", pil.size, (255, 255, 255, 255))
    # pil = Image.alpha_composite(_bg, pil).convert("RGB")
    
    img = pil.convert("L")
    arr = np.array(img, dtype=np.uint8)
    _, thr = cv.threshold(arr, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thr = Image.fromarray(thr).resize((size, size), Image.NEAREST)
    bits = (np.array(thr, dtype=np.uint8) > 127).astype(np.uint8)
    return bits

# =========================
# DWT helpers
# =========================
def dwt2_levels(img_2d, levels=DWT_LEVELS):
    return pywt.wavedec2(img_2d, wavelet=WAVELET_NAME, level=levels, mode=WAVELET_MODE)

def idwt2_levels(coeffs):
    return pywt.waverec2(coeffs, wavelet=WAVELET_NAME, mode=WAVELET_MODE)

def get_detail_bands_all_levels(coeffs):
    """
    Return a list of (level_idx, band_tag, band_array_ref) for detail bands only.
    band_tag ∈ {'H','V'} (no diagonal band).
    """
    out = []
    levels = len(coeffs) - 1
    for l in range(levels, 0, -1):
        cH, cV, cD = coeffs[levels - l + 1]
        out.append((l, 'H', cH))
        out.append((l, 'V', cV))
    return out

def set_detail_band(coeffs, level_idx, band_tag, new_arr):
    """Write back a specific detail band at a given level."""
    levels = len(coeffs) - 1
    i = levels - level_idx + 1
    cH, cV, cD = coeffs[i]
    if band_tag == 'H':
        coeffs[i] = (new_arr, cV, cD)
    elif band_tag == 'V':
        coeffs[i] = (cH, new_arr, cD)

# =========================
# DCT (8×8 blockwise) helpers
# =========================
def dct_blocks_8x8(arr2d):
    """Blockwise 8×8 DCT with edge padding to multiples of 8."""
    H, W = arr2d.shape
    H8 = (H + 7) // 8 * 8
    W8 = (W + 7) // 8 * 8
    padded = np.pad(arr2d, ((0, H8 - H), (0, W8 - W)), mode='edge').astype(np.float64)
    D = np.empty((H8, W8), dtype=np.float64)
    for i in range(0, H8, 8):
        for j in range(0, W8, 8):
            blk = padded[i:i+8, j:j+8]
            D[i:i+8, j:j+8] = dct(dct(blk.T, norm="ortho").T, norm="ortho")
    return D, H, W

def idct_blocks_8x8(D, H, W):
    """Inverse blockwise 8×8 DCT; crop back to original size."""
    H8, W8 = D.shape
    out = np.empty_like(D)
    for i in range(0, H8, 8):
        for j in range(0, W8, 8):
            blk = D[i:i+8, j:j+8]
            out[i:i+8, j:j+8] = idct(idct(blk.T, norm="ortho").T, norm="ortho")
    return out[:H, :W]

# =========================
# SVD-on-block helpers (inside an 8×8 DCT block)
# =========================
def svd_qim_embed_in_block(D8, bit, step, svd_idx=0):
    """Embed bit into the singular value of an 8x8 DCT block using QIM."""
    U, S, Vt = np.linalg.svd(D8, full_matrices=False)
    k = min(svd_idx, len(S) - 1)
    S[k] = qim_embed_scalar(S[k], int(bit), step)
    D8[:] = (U * S) @ Vt  # U @ diag(S) @ Vt

def svd_qim_llr_from_block(D8, step, svd_idx=0):
    """Extract the soft-decision score for a given bit."""
    _, S, _ = np.linalg.svd(D8, full_matrices=False)
    k = min(svd_idx, len(S) - 1)
    return qim_llr_scalar(S[k], step)

# =========================
# QIM primitives (scalar) on a single value
# =========================
def qim_embed_scalar(x, bit, step):
    """Quantize x into the center of the bit-labeled interval with step Δ=step."""
    q = np.floor(x / step)
    return (q + (0.25 if bit == 0 else 0.75)) * step

def qim_llr_scalar(x, step):
    """Soft-decision score: >0 means closer to bit=1; <0 closer to bit=0."""
    x /= step
    frac = x - np.floor(x)
    return abs(frac - 0.25) - abs(frac - 0.75)
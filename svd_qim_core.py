"""
SVD–QIM Watermarking (Color host, blind, no host resizing at embed)

Core idea
---------
1) Work in the Y (luma) channel for perceptual robustness.
2) Apply multi-level DWT; operate only on detail bands (default) for imperceptibility.
3) For each 8×8 DCT block in selected bands/tiles, take a mid-band patch,
   do SVD, and quantize ONE singular value (σ_k) via QIM (Σ-only embedding).
4) Use randomized block assignment (seeded by secret key + band/tile tags) and repetition
   + soft voting at extraction for robustness (e.g., compression and small local cutouts).

This file contains ONLY the SVD_QIM method (the original DCT-QIM branch is removed).
"""

import os
import math
import cv2
import numpy as np
from math import log10, sqrt
from PIL import Image
import pywt
from scipy.fftpack import dct, idct

# ===============================
# Utility: PSNR & BER
# ===============================

def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Peak-SNR between two uint8 images (same size)."""
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))

def bit_error_rate(true_bits, pred_bits, threshold=None, return_counts=False):
    """
    Compute BER between two binary (or gray) arrays/lists.
    If arrays are grayscale, they will be binarized by 0.5 (0..1) or 127 (0..255) unless threshold given.
    """
    a = np.asarray(true_bits)
    b = np.asarray(pred_bits)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    def binarize(x):
        x = np.asarray(x)
        thr = (0.5 if x.max() <= 1.0 else 127) if threshold is None else threshold
        return (x > thr).astype(np.uint8)

    a_bin = binarize(a)
    b_bin = binarize(b)
    errors = np.count_nonzero(a_bin ^ b_bin)
    N = a_bin.size
    ber = errors / N

    if not return_counts:
        return ber

    tp = np.count_nonzero((a_bin == 1) & (b_bin == 1))
    tn = np.count_nonzero((a_bin == 0) & (b_bin == 0))
    fp = np.count_nonzero((a_bin == 0) & (b_bin == 1))
    fn = np.count_nonzero((a_bin == 1) & (b_bin == 0))
    return ber, {
        "errors": errors, "total": N, "accuracy": 1.0 - ber,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

# ===============================
# I/O: Keep original size, YCbCr (we embed in Y only)
# ===============================

def load_rgb_keep_size(path: str):
    """
    Load image as RGB, convert to YCbCr, and return (Y_float64, Cb_u8, Cr_u8, (W,H)).
    We process Y for watermarking, then recombine with original Cb/Cr to save.
    """
    im = Image.open(path).convert("RGB")
    ycbcr = im.convert("YCbCr")
    y_luma = np.array(ycbcr.getchannel(0), dtype=np.float64)
    cb_chroma = np.array(ycbcr.getchannel(1), dtype=np.uint8)
    cr_chroma = np.array(ycbcr.getchannel(2), dtype=np.uint8)
    return y_luma, cb_chroma, cr_chroma, im.size  # (W,H)

def save_rgb_from_y(y_luma: np.ndarray, cb_chroma: np.ndarray, cr_chroma: np.ndarray, out_path: str):
    """
    Save RGB by merging a (possibly float) Y with original Cb/Cr.
    """
    y_u8 = np.rint(np.clip(y_luma, 0, 255)).astype(np.uint8)
    ycbcr = Image.merge(
        "YCbCr",
        (Image.fromarray(y_u8), Image.fromarray(cb_chroma), Image.fromarray(cr_chroma))
    )
    rgb = ycbcr.convert("RGB")
    rgb.save(out_path)

# ===============================
# DWT helpers
# ===============================

def dwt2(image_2d: np.ndarray, wavelet: str = 'haar', level: int = 1, mode: str = 'periodization'):
    return pywt.wavedec2(image_2d, wavelet=wavelet, level=level, mode=mode)

def idwt2(coeffs, wavelet: str = 'haar', mode: str = 'periodization'):
    return pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)

def wavelet_subband_map(coeffs):
    """
    Convert pywt 2D coeffs (level=1) to a dict: {"LL": arr, "LH": arr, "HL": arr, "HH": arr}
    """
    d = {"LL": coeffs[0]}
    lh, hl, hh = coeffs[1]
    d["LH"], d["HL"], d["HH"] = lh, hl, hh
    return d

def update_wavelet_subband(coeffs, band: str, new_arr: np.ndarray):
    """Write back a specific subband into coeffs (level=1)."""
    lh, hl, hh = coeffs[1]
    if band == "LL":
        coeffs[0] = new_arr
    elif band == "LH":
        coeffs[1] = (new_arr, hl, hh)
    elif band == "HL":
        coeffs[1] = (lh, new_arr, hh)
    elif band == "HH":
        coeffs[1] = (lh, hl, new_arr)

# ===============================
# Block DCT with padding to multiple of 8 (no host resize)
# ===============================

def dct2d_blocks_8x8(full_arr: np.ndarray):
    """
    Return DCT of the image (padded to multiples of 8), and original H,W for inverse crop.
    We apply 2D DCT per 8×8 block (JPEG-like).
    """
    H, W = full_arr.shape
    H8 = (H + 7) // 8 * 8
    W8 = (W + 7) // 8 * 8
    padded = np.pad(full_arr, ((0, H8 - H), (0, W8 - W)), mode='edge').astype(np.float64)
    D = np.empty((H8, W8), dtype=np.float64)
    for i in range(0, H8, 8):
        for j in range(0, W8, 8):
            blk = padded[i:i+8, j:j+8]
            D[i:i+8, j:j+8] = dct(dct(blk.T, norm="ortho").T, norm="ortho")
    return D, H, W

def idct2d_blocks_8x8(D: np.ndarray, H: int, W: int):
    """Inverse of dct2d_blocks_8x8; crop back to original H×W."""
    H8, W8 = D.shape
    rec = np.empty_like(D)
    for i in range(0, H8, 8):
        for j in range(0, W8, 8):
            blk = D[i:i+8, j:j+8]
            rec[i:i+8, j:j+8] = idct(idct(blk.T, norm="ortho").T, norm="ortho")
    return rec[:H, :W]

# ===============================
# QIM primitives (for Σ quantization inside SVD)
# ===============================

def embed_bit_by_qim(coeff_value: float, bit: int, step: float) -> float:
    """
    Scalar QIM embed of 1 bit into a single coefficient value.
    Places coeff into the center of the nearest bit-labeled interval of size 'step'.
    - bit=0 → interval centered at 0.25*step
    - bit=1 → interval centered at 0.75*step
    """
    q = np.floor(coeff_value / step)
    return (q + (0.25 if bit == 0 else 0.75)) * step

def qim_log_likelihood_ratio(coeff_value: float, step: float) -> float:
    """
    Soft decision: >0 means 'closer to bit=1', <0 means 'closer to bit=0'.
    Computed via distances to the 0.25*step and 0.75*step targets within the quantization cell.
    """
    x = coeff_value / step
    frac = x - np.floor(x)
    return abs(frac - 0.25) - abs(frac - 0.75)

# ===============================
# SVD-on-DCT-block helpers (Σ-only QIM)
# ===============================

def embed_bit_in_block_svd_qim(
    dct_block_8x8: np.ndarray,
    bit: int,
    step: float,
    svd_index: int = 0,
    patch_rows=(2, 6),
    patch_cols=(2, 6)
) -> None:
    """
    In-place modification of a mid-band patch inside an 8×8 DCT block via Σ-only QIM.

    Steps:
    1) Extract a mid-band patch from the 8×8 DCT block (rows 2..5, cols 2..5 by default).
    2) Compute SVD of the patch: P = U * diag(S) * V^T.
    3) Quantize ONE singular value S[svd_index] using QIM to embed 'bit'.
    4) Reconstruct the patch with the modified singular value and write it back.
    """
    r0, r1 = patch_rows
    c0, c1 = patch_cols
    patch = dct_block_8x8[r0:r1, c0:c1]
    U, S, Vt = np.linalg.svd(patch, full_matrices=False)
    k = min(svd_index, len(S) - 1)
    S[k] = embed_bit_by_qim(S[k], int(bit), step)
    patch_mod = (U * S) @ Vt  # equivalent to U @ diag(S) @ Vt
    dct_block_8x8[r0:r1, c0:c1] = patch_mod

def svd_qim_llr_for_block(
    dct_block_8x8: np.ndarray,
    step: float,
    svd_index: int = 0,
    patch_rows=(2, 6),
    patch_cols=(2, 6)
) -> float:
    """
    Return soft evidence (LLR) for the embedded bit from a single 8×8 block.
    """
    r0, r1 = patch_rows
    c0, c1 = patch_cols
    patch = dct_block_8x8[r0:r1, c0:c1]
    U, S, Vt = np.linalg.svd(patch, full_matrices=False)
    k = min(svd_index, len(S) - 1)
    return qim_log_likelihood_ratio(S[k], step)

# ===============================
# Tiling (for crop/small cutout robustness)
# ===============================

def compute_tile_slices(H: int, W: int, tiles=(2, 2)):
    """
    Split an array into a grid of tiles. Returns list of (rows_slice, cols_slice, tile_r, tile_c).
    We embed/extract bits across tiles with independent RNG shuffles, improving spatial spread.
    """
    tr, tc = tiles
    h_edges = [0] + [(H * r) // tr for r in range(1, tr)] + [H]
    w_edges = [0] + [(W * c) // tc for c in range(1, tc)] + [W]
    out = []
    for r in range(tr):
        for c in range(tc):
            r0, r1 = h_edges[r], h_edges[r + 1]
            c0, c1 = w_edges[c], w_edges[c + 1]
            out.append((slice(r0, r1), slice(c0, c1), r, c))
    return out

# ===============================
# Single-level SVD-QIM (kept for completeness)
# ===============================

BAND_TAG = {"LL": 101, "LH": 102, "HL": 103, "HH": 104}

def embed_bits_tiled_y_svd_qim(
    y_luma: np.ndarray,
    bits: list,
    secret_key: int,
    quant_step: float,
    svd_index: int = 0,
    svd_patch=((2, 6), (2, 6)),
    bands=("LL", "LH", "HL"),
    wavelet='haar',
    dwt_level=1,
    tiles=(2, 2),
    repeat=0
):
    """
    Embed bitstream into selected DWT subbands of Y using SVD-QIM on 8×8 DCT patches.
    """
    coeffs = dwt2(y_luma, wavelet, dwt_level)
    bands_map = wavelet_subband_map(coeffs)
    pr, pc = svd_patch

    for band in bands:
        sb = bands_map[band]
        for sl_r, sl_c, tile_r, tile_c in compute_tile_slices(*sb.shape, tiles=tiles):
            tile = sb[sl_r, sl_c]
            D, H0, W0 = dct2d_blocks_8x8(tile)
            H8, W8 = D.shape
            n_rows, n_cols = H8 // 8, W8 // 8
            n_blocks = n_rows * n_cols
            n_bits = len(bits)

            # Decide how many repetitions per bit (if repeat==0, fill all blocks)
            rep = repeat if repeat > 0 else max(1, n_blocks // n_bits)
            if n_bits * rep > n_blocks:
                rep = max(1, n_blocks // n_bits)

            # RNG shuffle for block positions (seeded by key+tile+band)
            seed = (int(secret_key) ^ (int(tile_r) << 8) ^ (int(tile_c) << 4) ^ BAND_TAG.get(band, 99)) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            idx = np.arange(n_blocks)
            rng.shuffle(idx)

            pos = 0
            for bit_idx, bit in enumerate(bits):
                for _ in range(rep):
                    if pos >= n_blocks:
                        break
                    k = int(idx[pos]); pos += 1
                    r = k // n_cols; c = k % n_cols
                    i, j = r * 8, c * 8
                    embed_bit_in_block_svd_qim(
                        D[i:i+8, j:j+8], int(bit), quant_step,
                        svd_index=svd_index, patch_rows=pr, patch_cols=pc
                    )

            sb[sl_r, sl_c] = idct2d_blocks_8x8(D, H0, W0)

        bands_map[band] = sb

    for band in bands:
        update_wavelet_subband(coeffs, band, bands_map[band])
    y_watermarked = idwt2(coeffs, wavelet)
    return y_watermarked

def extract_bits_tiled_y_svd_qim(
    y_luma: np.ndarray,
    n_bits: int,
    secret_key: int,
    quant_step: float,
    svd_index: int = 0,
    svd_patch=((2, 6), (2, 6)),
    bands=("LL", "LH", "HL"),
    wavelet='haar',
    dwt_level=1,
    tiles=(2, 2),
):
    """
    Extract bitstream using soft voting over randomized, repeated block positions.
    Returns a list of {0,1}.
    """
    coeffs = dwt2(y_luma, wavelet, dwt_level)
    bands_map = wavelet_subband_map(coeffs)
    soft_sum = np.zeros(n_bits, dtype=np.float64)
    pr, pc = svd_patch

    for band in bands:
        sb = bands_map[band]
        for sl_r, sl_c, tile_r, tile_c in compute_tile_slices(*sb.shape, tiles=tiles):
            tile = sb[sl_r, sl_c]
            D, H0, W0 = dct2d_blocks_8x8(tile)
            H8, W8 = D.shape
            n_rows, n_cols = H8 // 8, W8 // 8
            n_blocks = n_rows * n_cols

            # How many blocks per bit (mirror of embed when repeat=0)
            rep = max(1, n_blocks // n_bits)

            seed = (int(secret_key) ^ (int(tile_r) << 8) ^ (int(tile_c) << 4) ^ BAND_TAG.get(band, 99)) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            idx = np.arange(n_blocks)
            rng.shuffle(idx)

            pos = 0
            for bit_idx in range(n_bits):
                score = 0.0
                for _ in range(rep):
                    if pos >= n_blocks:
                        break
                    k = int(idx[pos]); pos += 1
                    r = k // n_cols; c = k % n_cols
                    i, j = r * 8, c * 8
                    score += svd_qim_llr_for_block(
                        D[i:i+8, j:j+8], quant_step,
                        svd_index=svd_index, patch_rows=pr, patch_cols=pc
                    )
                soft_sum[bit_idx] += score

    # Soft decision across all bands/tiles:
    out_bits = (soft_sum >= 0).astype(np.uint8).tolist()
    return out_bits

# ===============================
# Multi-level variants (LEVEL>=1)
# ===============================

def bands_all_levels(coeffs, use_LL=True, use_HV=True, use_D=False):
    """
    Create a list of (level_index, band_name, array_view) across levels.
    Level indexing: L is deepest. For each level, return H (horizontal), V (vertical), D (diagonal) as requested.
    """
    out = []
    L = len(coeffs) - 1
    if use_LL:
        out.append((L, 'LL', coeffs[0]))
    for l in range(L, 0, -1):
        (cH, cV, cD) = coeffs[L - l + 1]
        if use_HV:
            out.append((l, 'H', cH))
            out.append((l, 'V', cV))
        if use_D:
            out.append((l, 'D', cD))
    return out

def set_band_at_level(coeffs, level_index, band_name, new_arr):
    """Write back band at specific level."""
    L = len(coeffs) - 1
    if band_name == 'LL':
        coeffs[0] = new_arr; return
    idx = L - level_index + 1
    cH, cV, cD = coeffs[idx]
    if band_name == 'H':
        coeffs[idx] = (new_arr, cV, cD)
    elif band_name == 'V':
        coeffs[idx] = (cH, new_arr, cD)
    elif band_name == 'D':
        coeffs[idx] = (cH, cV, new_arr)

def embed_bits_y_multilevel_svd_qim(
    y_luma: np.ndarray,
    bits: list,
    secret_key: int,
    quant_step: float,
    svd_index: int = 0,
    svd_patch=((2, 6), (2, 6)),
    wavelet='haar',
    dwt_levels=2,
    tiles=(2, 2),
    repeat=0,
    include_LL=False,
    include_D=False
):
    """
    Multi-level version: pad Y to match DWT stride, embed across H/V (and optionally LL/D) at all levels.
    """
    # Pad so that after multi-level transforms, we still align to multiples of 8 for DCT blocks
    stride = 1 << dwt_levels
    mult = stride if stride >= 8 else 8
    H0, W0 = y_luma.shape
    Hm = (H0 + mult - 1) // mult * mult
    Wm = (W0 + mult - 1) // mult * mult
    y_pad = np.pad(y_luma, ((0, Hm - H0), (0, Wm - W0)), mode='edge')

    coeffs = dwt2(y_pad, wavelet, dwt_levels)
    target_bands = bands_all_levels(coeffs, use_LL=include_LL, use_HV=True, use_D=include_D)
    pr, pc = svd_patch

    for lvl, bname, sb in target_bands:
        for sl_r, sl_c, tile_r, tile_c in compute_tile_slices(*sb.shape, tiles=tiles):
            tile = sb[sl_r, sl_c]
            D, Ht, Wt = dct2d_blocks_8x8(tile)
            H8, W8 = D.shape
            n_rows, n_cols = H8 // 8, W8 // 8
            n_blocks = n_rows * n_cols
            n_bits = len(bits)

            rep = repeat if repeat > 0 else max(1, n_blocks // n_bits)
            if n_bits * rep > n_blocks:
                rep = max(1, n_blocks // n_bits)

            tag = {'LL': 201, 'H': 202, 'V': 203, 'D': 204}[bname]
            seed = (int(secret_key) ^ (int(tile_r) << 9) ^ (int(tile_c) << 5) ^ (lvl << 1) ^ tag) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            idx = np.arange(n_blocks); rng.shuffle(idx)

            pos = 0
            for bit_idx, bit in enumerate(bits):
                for _ in range(rep):
                    if pos >= n_blocks:
                        break
                    k = int(idx[pos]); pos += 1
                    r = k // n_cols; c = k % n_cols
                    i, j = r * 8, c * 8
                    embed_bit_in_block_svd_qim(
                        D[i:i+8, j:j+8], int(bit), quant_step,
                        svd_index=svd_index, patch_rows=pr, patch_cols=pc
                    )

            sb[sl_r, sl_c] = idct2d_blocks_8x8(D, Ht, Wt)
        set_band_at_level(coeffs, lvl, bname, sb)

    y_wm = idwt2(coeffs, wavelet)
    return y_wm[:H0, :W0]

def extract_bits_y_multilevel_svd_qim(
    y_luma: np.ndarray,
    n_bits: int,
    secret_key: int,
    quant_step: float,
    svd_index: int = 0,
    svd_patch=((2, 6), (2, 6)),
    wavelet='haar',
    dwt_levels=2,
    tiles=(2, 2),
    include_LL=False,
    include_D=False
):
    """
    Multi-level extraction with soft voting across all used bands/tiles/levels.
    """
    stride = 1 << dwt_levels
    mult = stride if stride >= 8 else 8
    H0, W0 = y_luma.shape
    Hm = (H0 + mult - 1) // mult * mult
    Wm = (W0 + mult - 1) // mult * mult
    y_pad = np.pad(y_luma, ((0, Hm - H0), (0, Wm - W0)), mode='edge')

    coeffs = dwt2(y_pad, wavelet, dwt_levels)
    soft_sum = np.zeros(n_bits, dtype=np.float64)
    target_bands = bands_all_levels(coeffs, use_LL=include_LL, use_HV=True, use_D=include_D)
    pr, pc = svd_patch

    for lvl, bname, sb in target_bands:
        for sl_r, sl_c, tile_r, tile_c in compute_tile_slices(*sb.shape, tiles=tiles):
            tile = sb[sl_r, sl_c]
            D, Ht, Wt = dct2d_blocks_8x8(tile)
            H8, W8 = D.shape
            n_rows, n_cols = H8 // 8, W8 // 8
            n_blocks = n_rows * n_cols

            rep = max(1, n_blocks // n_bits)
            tag = {'LL': 201, 'H': 202, 'V': 203, 'D': 204}[bname]
            seed = (int(secret_key) ^ (int(tile_r) << 9) ^ (int(tile_c) << 5) ^ (lvl << 1) ^ tag) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            idx = np.arange(n_blocks); rng.shuffle(idx)

            pos = 0
            for bit_idx in range(n_bits):
                score = 0.0
                for _ in range(rep):
                    if pos >= n_blocks:
                        break
                    k = int(idx[pos]); pos += 1
                    r = k // n_cols; c = k % n_cols
                    i, j = r * 8, c * 8
                    score += svd_qim_llr_for_block(
                        D[i:i+8, j:j+8], quant_step,
                        svd_index=svd_index, patch_rows=pr, patch_cols=pc
                    )
                soft_sum[bit_idx] += score

    out_bits = (soft_sum >= 0).astype(np.uint8).tolist()
    return out_bits

# ===============================
# Batch wrappers (Y-channel lists)
# ===============================
def embed_bits_y_multilevel_svd_qim_batch(
    y_lumas: list,
    bits: list,
    secret_key: int,
    quant_step: float,
    svd_index: int = 0,
    svd_patch=((2, 6), (2, 6)),
    wavelet='haar',
    dwt_levels: int = 2,
    tiles=(2, 2),
    repeat: int = 0,
    include_LL: bool = False,
    include_D: bool = False
):
    """Return list of Y_wm arrays, one per input Y."""
    out = []
    for y in y_lumas:
        y_wm = embed_bits_y_multilevel_svd_qim(
            y_luma=y,
            bits=bits,
            secret_key=secret_key,
            quant_step=quant_step,
            svd_index=svd_index,
            svd_patch=svd_patch,
            wavelet=wavelet,
            dwt_levels=dwt_levels,
            tiles=tiles,
            repeat=repeat,
            include_LL=include_LL,
            include_D=include_D,
        )
        out.append(y_wm)
    return out

def extract_bits_y_multilevel_svd_qim_batch(
    y_lumas: list,
    n_bits: int,
    secret_key: int,
    quant_step: float,
    svd_index: int = 0,
    svd_patch=((2, 6), (2, 6)),
    wavelet='haar',
    dwt_levels: int = 2,
    tiles=(2, 2),
    include_LL: bool = False,
    include_D: bool = False
):
    """Return list[ list[int] ] of extracted bits (one per Y)."""
    out_bits = []
    for y in y_lumas:
        bits = extract_bits_y_multilevel_svd_qim(
            y_luma=y,
            n_bits=n_bits,
            secret_key=secret_key,
            quant_step=quant_step,
            svd_index=svd_index,
            svd_patch=svd_patch,
            wavelet=wavelet,
            dwt_levels=dwt_levels,
            tiles=tiles,
            include_LL=include_LL,
            include_D=include_D,
        )
        out_bits.append(bits)
    return out_bits

# ===============================
# Attack helpers (for testing)
# ===============================

def save_jpeg(input_path: str, out_path: str, quality: int = 75):
    Image.open(input_path).convert("RGB").save(out_path, quality=quality, subsampling=0, optimize=False)

def save_small_random_cutout(
    input_path: str,
    out_path: str,
    area_ratio: float = 0.01,
    num_patches: int = 1,
    shape: str = "rect",          # 'rect' or 'circle'
    fill: str = "noise",          # 'noise'|'black'|'avg'|'blur'|'inpaint'
    blur_kernel: int = 11,
    seed: int | None = None
):
    """
    Remove one or more small regions from the image while keeping the same size.

    This simulates 'tiny defects' or stickers/dust (small cutout) without global resizing/cropping.
    """
    im = Image.open(input_path).convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    H, W, C = arr.shape
    rng = np.random.default_rng(seed)

    def mask_rect(x0, y0, w, h):
        m = np.zeros((H, W), dtype=np.uint8)
        m[y0:y0+h, x0:x0+w] = 255
        return m

    def mask_circle(x0, y0, w, h):
        m = np.zeros((H, W), dtype=np.uint8)
        cy, cx = y0 + h // 2, x0 + w // 2
        r = int(0.5 * max(w, h))
        yy, xx = np.ogrid[:H, :W]
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        m[circle] = 255
        return m

    for _ in range(max(1, int(num_patches))):
        # Patch size ~ sqrt(area_ratio) * (W,H) → area ≈ area_ratio * W * H
        w = max(1, int(round(np.sqrt(area_ratio) * W)))
        h = max(1, int(round(np.sqrt(area_ratio) * H)))
        x0 = int(rng.integers(0, max(1, W - w)))
        y0 = int(rng.integers(0, max(1, H - h)))
        mask = mask_rect(x0, y0, w, h) if shape == "rect" else mask_circle(x0, y0, w, h)

        if fill == "black":
            arr[mask == 255] = 0
        elif fill == "avg":
            mean = arr.reshape(-1, C).mean(axis=0).astype(np.uint8)
            arr[mask == 255] = mean
        elif fill == "noise":
            noise = rng.integers(0, 256, size=(H, W, C), dtype=np.uint8)
            arr[mask == 255] = noise[mask == 255]
        elif fill == "blur":
            k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            blurred = cv2.GaussianBlur(arr, (k, k), 0)
            arr[mask == 255] = blurred[mask == 255]
        elif fill == "inpaint":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            bgr = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("fill must be one of {'noise','black','avg','blur','inpaint'}")

    Image.fromarray(arr).save(out_path)
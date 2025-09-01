import numpy as np
from PIL import Image
import utilities
from parameters import WATERMARK_SIZE, DWT_LEVELS, QIM_STEP

# =========================
# Embedding
# =========================
def embed_watermark(y_luma, wm_bits_2d):
    """
    Embed size×size binary watermark into Y using deterministic round-robin mapping.
    - Never touches LL subbands.
    - No randomness: block k carries bit (k % n_bits).
    """
    H0, W0 = y_luma.shape
    if not (512 <= H0 <= 1080 and 512 <= W0 <= 1080):
        raise ValueError("Host image must be within 512–1080 px on BOTH height and width.")

    # Pad so DWT levels and 8×8 DCT align cleanly
    stride = 1 << DWT_LEVELS
    mult = stride if stride >= 8 else 8
    Hp = (H0 + mult - 1) // mult * mult
    Wp = (W0 + mult - 1) // mult * mult
    y_pad = np.pad(y_luma, ((0, Hp - H0), (0, Wp - W0)), mode='edge')

    # DWT
    coeffs = utilities.dwt2_levels(y_pad, DWT_LEVELS)
    detail_bands = utilities.get_detail_bands_all_levels(coeffs)

    wm_bits = wm_bits_2d.astype(np.uint8).ravel()
    n_bits  = wm_bits.size

    for lvl, tag, band in detail_bands:
        # DCT over the entire band (single region)
        D, Hb, Wb = utilities.dct_blocks_8x8(band)
        H8, W8 = D.shape
        n_rows, n_cols = H8 // 8, W8 // 8
        n_blocks = n_rows * n_cols

        # Deterministic round-robin mapping: block k → bit (k % n_bits)
        assign = np.arange(n_blocks, dtype=np.int32) % n_bits

        # Embed per-block
        for k in range(n_blocks):
            bit_index = int(assign[k])
            r = k // n_cols; c = k % n_cols
            i, j = r * 8, c * 8
            utilities.svd_qim_embed_in_block(
                D[i:i+8, j:j+8],
                int(wm_bits[bit_index]),
                QIM_STEP
            )

        # Write back the modified band
        band[:] = utilities.idct_blocks_8x8(D, Hb, Wb)
        utilities.set_detail_band(coeffs, lvl, tag, band)

    # Inverse DWT and crop back to original size
    y_wm_pad = utilities.idwt2_levels(coeffs)
    return y_wm_pad[:H0, :W0]

# =========================
# Extraction (Main)
# =========================
def extract_watermark(y_luma, wm_size=WATERMARK_SIZE, return_score=False):
    """
    Recreate the same deterministic round-robin mapping and do soft voting.
    """
    H0, W0 = y_luma.shape
    stride = 1 << DWT_LEVELS
    mult = stride if stride >= 8 else 8
    Hp = (H0 + mult - 1) // mult * mult
    Wp = (W0 + mult - 1) // mult * mult
    y_pad = np.pad(y_luma, ((0, Hp - H0), (0, Wp - W0)), mode='edge')

    coeffs = utilities.dwt2_levels(y_pad, DWT_LEVELS)
    detail_bands = utilities.get_detail_bands_all_levels(coeffs)

    n_bits = wm_size * wm_size
    soft_sum = np.zeros(n_bits, dtype=np.float64)

    for lvl, tag, band in detail_bands:
        D, Hb, Wb = utilities.dct_blocks_8x8(band)
        H8, W8 = D.shape
        n_rows, n_cols = H8 // 8, W8 // 8
        n_blocks = n_rows * n_cols

        # Same round-robin mapping
        assign = np.arange(n_blocks, dtype=np.int32) % n_bits

        for k in range(n_blocks):
            r = k // n_cols; c = k % n_cols
            i, j = r * 8, c * 8
            llr = utilities.svd_qim_llr_from_block(
                D[i:i+8, j:j+8],
                QIM_STEP
            )
            soft_sum[int(assign[k])] += llr

    bits = (soft_sum >= 0).astype(np.uint8)

    # for rotation enhancement
    if return_score:
        return bits.reshape(wm_size, wm_size), float(np.sum(np.abs(soft_sum)))
    
    return bits.reshape(wm_size, wm_size)

# =======================================
# Extraction (Enhanced for small rotation)
# =======================================
def extract_watermark_enhanced(y_luma, wm_size=WATERMARK_SIZE, angle_range=(-5, 5), step=0.5):
    """
    Brute-force small rotation search around 0°.
    Accepts y_luma as np.ndarray (H×W) or PIL.Image.
    """
    y_img = utilities.to_pil_luma(y_luma)

    best_bits, best_score = None, -np.inf
    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, step, dtype=float)
    for ang in angles:
        # rotate expects PIL; extractor likely expects np.ndarray → convert back
        rot_img = y_img.rotate(float(ang), resample=Image.BICUBIC, expand=False)
        rot_arr = np.asarray(rot_img)
        bits, score = extract_watermark(rot_arr, wm_size, return_score=True)
        if score > best_score:
            best_score, best_bits = score, bits
    return best_bits
import numpy as np
from PIL import Image
import utils
from parameters import WATERMARK_SIZE, DWT_LEVELS, QIM_STEP

# =========================
# Embedding (Y channel only)
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
    coeffs = utils.dwt2_levels(y_pad, DWT_LEVELS)
    detail_bands = utils.get_detail_bands_all_levels(coeffs)

    wm_bits = wm_bits_2d.astype(np.uint8).ravel()
    n_bits  = wm_bits.size

    for lvl, tag, band in detail_bands:
        # DCT over the entire band (single region)
        D, Hb, Wb = utils.dct_blocks_8x8(band)
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
            utils.svd_qim_embed_in_block(
                D[i:i+8, j:j+8],
                int(wm_bits[bit_index]),
                QIM_STEP
            )

        # Write back the modified band
        band[:] = utils.idct_blocks_8x8(D, Hb, Wb)
        utils.set_detail_band(coeffs, lvl, tag, band)

    # Inverse DWT and crop back to original size
    y_wm_pad = utils.idwt2_levels(coeffs)
    return y_wm_pad[:H0, :W0]

# =========================
# Extraction (blind)
# =========================
def extract_watermark(y_luma, wm_size=WATERMARK_SIZE):
    """
    Recreate the same deterministic round-robin mapping and do soft voting.
    """
    H0, W0 = y_luma.shape
    stride = 1 << DWT_LEVELS
    mult = stride if stride >= 8 else 8
    Hp = (H0 + mult - 1) // mult * mult
    Wp = (W0 + mult - 1) // mult * mult
    y_pad = np.pad(y_luma, ((0, Hp - H0), (0, Wp - W0)), mode='edge')

    coeffs = utils.dwt2_levels(y_pad, DWT_LEVELS)
    detail_bands = utils.get_detail_bands_all_levels(coeffs)

    n_bits = wm_size * wm_size
    soft_sum = np.zeros(n_bits, dtype=np.float64)

    for lvl, tag, band in detail_bands:
        D, Hb, Wb = utils.dct_blocks_8x8(band)
        H8, W8 = D.shape
        n_rows, n_cols = H8 // 8, W8 // 8
        n_blocks = n_rows * n_cols

        # Same round-robin mapping
        assign = np.arange(n_blocks, dtype=np.int32) % n_bits

        for k in range(n_blocks):
            r = k // n_cols; c = k % n_cols
            i, j = r * 8, c * 8
            llr = utils.svd_qim_llr_from_block(
                D[i:i+8, j:j+8],
                QIM_STEP
            )
            soft_sum[int(assign[k])] += llr

    bits = (soft_sum >= 0).astype(np.uint8)
    return bits.reshape(wm_size, wm_size)

# =========================
# High-level convenience I/O
# =========================
# def embed_watermark(host_path, wm_path, out_path="watermarked.png"):
#     """Embed watermark image into host image file and save the watermarked RGB."""
#     rgb = Image.open(host_path).convert("RGB")
#     Y, Cb, Cr = utils.to_ycbcr_arrays(rgb)
#     wm_bits = utils.prepare_watermark_bits(wm_path, WATERMARK_SIZE)
#     Y_wm = embed_watermark_y(Y, wm_bits)
#     out_rgb = utils.from_ycbcr_arrays(Y_wm, Cb, Cr)
#     out_rgb.save(out_path)
#     return np.array(rgb, dtype=np.uint8), np.array(out_rgb, dtype=np.uint8), wm_bits

# def extract_watermark(image_path, wm_size=WATERMARK_SIZE):
#     """Extract watermark bits (wm_size×wm_size) from a watermarked/attacked image file."""
#     rgb = Image.open(image_path).convert("RGB")
#     Y, _, _ = utils.to_ycbcr_arrays(rgb)
#     wm_bits = extract_watermark_y(Y, wm_size)
#     return wm_bits
import io
import math
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import pywt

# ---- bring in your core functions (unchanged) ----
from svd_qim_core import (
    psnr, bit_error_rate,
    embed_bits_y_multilevel_svd_qim, extract_bits_y_multilevel_svd_qim,
    bands_all_levels,  # for capacity calc
)
from svd_qim_core import dwt2 as core_dwt2  # to match your wavelet settings
from svd_qim_core import compute_tile_slices  # same tiling logic as core

# ============================================
# Constants
# ============================================
WATERMARK_N = 32  # fixed 32x32
DEFAULT_SECRET = 1234567890

# ============================================
# Small helpers
# ============================================
def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)

def np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

def rgb_to_ycbcr_arrays(im_rgb: Image.Image):
    ycbcr = im_rgb.convert("YCbCr")
    y = np.array(ycbcr.getchannel(0), dtype=np.float64)
    cb = np.array(ycbcr.getchannel(1), dtype=np.uint8)
    cr = np.array(ycbcr.getchannel(2), dtype=np.uint8)
    return y, cb, cr

def ycbcr_arrays_to_rgb(y_luma: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> Image.Image:
    y_u8 = np.rint(np.clip(y_luma, 0, 255)).astype(np.uint8)
    ycbcr_img = Image.merge("YCbCr", (Image.fromarray(y_u8), Image.fromarray(cb), Image.fromarray(cr)))
    return ycbcr_img.convert("RGB")

def make_binary_watermark_from_pil(wm_pil: Image.Image, size=WATERMARK_N):
    wm_gray = wm_pil.convert("L").resize((size, size), Image.LANCZOS)
    wm_bin = (np.array(wm_gray, dtype=np.uint8) > 127).astype(np.uint8)
    return wm_bin  # shape (N,N) in {0,1}

def default_host(H=720, W=960):
    x = np.linspace(0, 1, W, dtype=np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    r = (xv * 255).astype(np.uint8)
    g = (yv * 255).astype(np.uint8)
    b = ((1 - 0.5 * xv - 0.5 * yv) * 255).clip(0, 255).astype(np.uint8)
    return np_rgb_to_pil(np.dstack([r, g, b]))

def np_image_download_button(img_pil: Image.Image, label: str, filename: str):
    buf = io.BytesIO()
    fmt = filename.split(".")[-1].upper()
    img_pil.save(buf, format="PNG" if fmt == "PNG" else fmt)
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/" + filename.split(".")[-1])

# In-memory small cutout (like your helper but no disk I/O)
def small_random_cutout_mem(
    in_rgb: np.ndarray,
    area_ratio=0.001,
    num_patches=50,
    shape="rect",
    fill="noise",
    blur_kernel=11,
    seed=None,
):
    arr = in_rgb.copy()
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

    for _ in range(max(1, int(numpatches := num_patches))):
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
        else:
            raise ValueError("fill must be one of {'noise','black','avg','blur'}")
    return arr

# ============================================
# Capacity & minimum-size logic
# ============================================
def count_blocks_for_subband_tile(h, w):
    # After DCT padding to multiples of 8, blocks = ceil(h/8) * ceil(w/8)
    return math.ceil(h / 8) * math.ceil(w / 8)

def total_capacity_for_shape(H, W, wavelet, dwt_levels, tiles, include_LL=False, include_D=False):
    # Build dummy array to get exact pywt shapes with 'periodization'
    dummy = np.zeros((H, W), dtype=np.float64)
    coeffs = pywt.wavedec2(dummy, wavelet=wavelet, level=dwt_levels, mode="periodization")
    target = bands_all_levels(coeffs, use_LL=include_LL, use_HV=True, use_D=include_D)

    total = 0
    for _, _, sb in target:
        h, w = sb.shape
        # identical tiling logic as core
        for sl_r, sl_c, _, _ in compute_tile_slices(h, w, tiles=tiles):
            th = sl_r.stop - sl_r.start
            tw = sl_c.stop - sl_c.start
            total += count_blocks_for_subband_tile(th, tw)
    return total

def recommend_min_square_size(n_bits, wavelet, dwt_levels, tiles, include_LL, include_D):
    # Find minimal N such that total capacity >= n_bits
    N = 64  # start reasonable
    for N in range(64, 4096 + 1, 8):
        cap = total_capacity_for_shape(N, N, wavelet, dwt_levels, tiles, include_LL, include_D)
        if cap >= n_bits:
            return N, cap
    return 4096, total_capacity_for_shape(4096, 4096, wavelet, dwt_levels, tiles, include_LL, include_D)

# ============================================
# Streamlit UI
# ============================================
st.set_page_config(page_title="SVD–QIM Watermarking Demo", layout="wide")
st.title("🔐 SVD–QIM Watermarking (DWT–DCT–SVD, Σ-only QIM)")

with st.sidebar:
    st.subheader("Parameters")
    st.caption("These must match between Embed and Extract.")
    secret_key  = st.number_input("Secret key (int)", value=DEFAULT_SECRET, step=1)
    quant_step  = st.slider("QIM quantization step (σₖ)", 10.0, 120.0, 80.0, step=1.0)
    wavelet     = st.selectbox("Wavelet", ["haar","db2","db4","sym4","coif1"], index=0)
    dwt_levels  = st.slider("DWT levels", 1, 4, 4, step=1)
    tiles_r     = st.selectbox("Tiles (rows)", [1,2,3,4], index=1)
    tiles_c     = st.selectbox("Tiles (cols)", [1,2,3,4], index=1)
    include_LL  = st.checkbox("Include LL band", value=False)
    include_D   = st.checkbox("Include Diagonal band (D)", value=False)

    st.subheader("SVD patch")
    svd_index   = st.selectbox("Singular value index k (0 = largest)", [0,1,2,3], index=0)
    r0 = st.slider("Patch row start", 0, 7, 2); r1 = st.slider("Patch row end (exclusive)", r0+1, 8, 6)
    c0 = st.slider("Patch col start", 0, 7, 2); c1 = st.slider("Patch col end (exclusive)", c0+1, 8, 6)

tabs = st.tabs(["Embed", "Extract", "Attack"])

# Keep session state
if "watermarked_rgb" not in st.session_state:
    st.session_state["watermarked_rgb"] = None
if "last_wm_bits" not in st.session_state:
    st.session_state["last_wm_bits"] = None  # the bits actually embedded
if "ref_wm_bits" not in st.session_state:
    st.session_state["ref_wm_bits"] = None   # user-provided original watermark for BER (Extract tab)

# ==========================================================
# TAB 1: EMBED
# ==========================================================
with tabs[0]:
    st.header("Embed")
    host_file = st.file_uploader("Host image (RGB). If none, a demo gradient will be used.", type=["png","jpg","jpeg","webp"], key="host_upl")
    wm_file   = st.file_uploader("Watermark image (will be binarized to fixed 32×32)", type=["png","jpg","jpeg","webp"], key="wm_upl")

    # Load host
    if host_file:
        host_pil = Image.open(host_file).convert("RGB")
    else:
        host_pil = default_host()
        st.info("No host uploaded — using a colorful synthetic image.")

    H, W = host_pil.size[1], host_pil.size[0]

    # Show recommended minimum size (for current params)
    n_bits = WATERMARK_N * WATERMARK_N
    recN, capN = recommend_min_square_size(
        n_bits=n_bits,
        wavelet=wavelet,
        dwt_levels=int(dwt_levels),
        tiles=(int(tiles_r), int(tiles_c)),
        include_LL=bool(include_LL),
        include_D=bool(include_D),
    )
    current_capacity = total_capacity_for_shape(
        H, W, wavelet, int(dwt_levels), (int(tiles_r), int(tiles_c)), bool(include_LL), bool(include_D)
    )
    st.markdown(
        f"**Recommended minimum host size (square)** for these settings: **{recN}×{recN}** "
        f"(capacity≈{capN} blocks for {n_bits} bits).  \n"
        f"Your image: **{W}×{H}**, capacity≈**{current_capacity}** blocks."
    )
    if current_capacity < n_bits:
        st.warning("Capacity is lower than 32×32=1024 bits in at least one tile/band; "
                   "embedding will still run but repetition/coverage per bit may be uneven. "
                   "Use a larger image or more tiles/bands/levels.")

    # Prepare watermark bits
    if wm_file:
        wm_pil = Image.open(wm_file)
        wm_bin = make_binary_watermark_from_pil(wm_pil, size=WATERMARK_N)
    else:
        st.info("Please upload a watermark image (any size); it will be binarized to 32×32.")
        wm_bin = None

    # Show inputs
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.markdown("**Host**")
        st.image(host_pil, use_column_width=True)
    with col_in2:
        st.markdown("**Watermark (fixed 32×32, binary preview)**")
        if wm_bin is not None:
            st.image((wm_bin * 255).astype(np.uint8), clamp=True, width=256)
        else:
            st.write("—")

    if st.button("🔧 Embed Watermark"):
        if wm_bin is None:
            st.warning("Upload a watermark first.")
        else:
            wm_bits = wm_bin.reshape(-1).tolist()
            y_host, cb_host, cr_host = rgb_to_ycbcr_arrays(host_pil)
            y_wm = embed_bits_y_multilevel_svd_qim(
                y_luma=y_host,
                bits=wm_bits,
                secret_key=int(secret_key),
                quant_step=float(quant_step),
                svd_index=int(svd_index),
                svd_patch=((r0, r1), (c0, c1)),
                wavelet=wavelet,
                dwt_levels=int(dwt_levels),
                tiles=(int(tiles_r), int(tiles_c)),
                repeat=0,
                include_LL=bool(include_LL),
                include_D=bool(include_D),
            )
            wm_rgb_pil = ycbcr_arrays_to_rgb(y_wm, cb_host, cr_host)
            wm_rgb = pil_to_np_rgb(wm_rgb_pil)
            host_rgb = pil_to_np_rgb(host_pil)

            st.session_state["watermarked_rgb"] = wm_rgb
            st.session_state["last_wm_bits"] = wm_bits  # store what we actually embedded

            # Show side-by-side
            col_a, col_b = st.columns(2, gap="large")
            with col_a:
                st.markdown("**Original**")
                st.image(host_pil, use_column_width=True)
            with col_b:
                st.markdown("**Watermarked**")
                st.image(wm_rgb_pil, use_column_width=True)

            # PSNR below
            psnr_val = psnr(host_rgb, wm_rgb)
            st.metric("PSNR (Original vs Watermarked)", f"{psnr_val:.2f} dB")

            # Download
            np_image_download_button(wm_rgb_pil, "⬇️ Download watermarked", "watermarked.png")

# ==========================================================
# TAB 2: EXTRACT
# ==========================================================
with tabs[1]:
    st.header("Extract")
    st.caption("Upload only the watermarked image for extraction. "
               "Upload the original watermark (32×32) if you want BER.")

    attacked_file = st.file_uploader("Watermarked image to extract from", type=["png","jpg","jpeg","webp"], key="attacked_upl")
    ori_wm_file   = st.file_uploader("Original watermark image (for BER, optional)", type=["png","jpg","jpeg","webp"], key="oriwm_upl")

    if st.button("🔎 Extract Watermark"):
        if attacked_file is None:
            st.warning("Please upload a watermarked image.")
        else:
            src_pil = Image.open(attacked_file).convert("RGB")
            y_src, _, _ = rgb_to_ycbcr_arrays(src_pil)
            bits_out = extract_bits_y_multilevel_svd_qim(
                y_luma=y_src,
                n_bits=WATERMARK_N * WATERMARK_N,
                secret_key=int(secret_key),
                quant_step=float(quant_step),
                svd_index=int(svd_index),
                svd_patch=((r0, r1), (c0, c1)),
                wavelet=wavelet,
                dwt_levels=int(dwt_levels),
                tiles=(int(tiles_r), int(tiles_c)),
                include_LL=bool(include_LL),
                include_D=bool(include_D),
            )
            wm_rec = np.array(bits_out, dtype=np.uint8).reshape(WATERMARK_N, WATERMARK_N)
            st.markdown("**Extracted watermark**")
            st.image((wm_rec * 255).astype(np.uint8), clamp=True, width=256)

            # If original watermark is provided, compute BER
            if ori_wm_file:
                ori_wm_pil = Image.open(ori_wm_file)
                ori_wm_bin = make_binary_watermark_from_pil(ori_wm_pil, size=WATERMARK_N)
                ref_bits = ori_wm_bin.reshape(-1).tolist()
                ber_val, stats = bit_error_rate(ref_bits, bits_out, return_counts=True)
                st.metric("BER", f"{ber_val:.4f}")
                st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")
            else:
                st.info("Upload the original 32×32 watermark image above to compute BER.")

# ==========================================================
# TAB 3: ATTACK
# ==========================================================
with tabs[2]:
    st.header("Attack")
    if st.session_state["watermarked_rgb"] is None:
        st.info("Please embed first in the **Embed** tab to run built-in attacks here.")
    else:
        wm_rgb = st.session_state["watermarked_rgb"]
        sub = st.tabs(["JPEG Compression", "Small Cutouts"])

        # JPEG Attack
        with sub[0]:
            q = st.slider("JPEG quality", 10, 95, 75, step=1, key="jpegq")
            if st.button("Run JPEG Attack"):
                buf = io.BytesIO()
                np_rgb_to_pil(wm_rgb).save(buf, format="JPEG", quality=q, subsampling=0, optimize=False)
                buf.seek(0)
                atk_pil = Image.open(buf).convert("RGB")
                st.image(atk_pil, caption=f"JPEG q={q}", use_column_width=True)

                # Extract after attack
                y_atk, _, _ = rgb_to_ycbcr_arrays(atk_pil)
                bits_atk = extract_bits_y_multilevel_svd_qim(
                    y_luma=y_atk,
                    n_bits=WATERMARK_N * WATERMARK_N,
                    secret_key=int(secret_key),
                    quant_step=float(quant_step),
                    svd_index=int(svd_index),
                    svd_patch=((r0, r1), (c0, c1)),
                    wavelet=wavelet,
                    dwt_levels=int(dwt_levels),
                    tiles=(int(tiles_r), int(tiles_c)),
                    include_LL=bool(include_LL),
                    include_D=bool(include_D),
                )
                wm_atk = np.array(bits_atk, dtype=np.uint8).reshape(WATERMARK_N, WATERMARK_N)

                col5, col6 = st.columns(2)
                with col5:
                    st.markdown("**Extracted (JPEG)**")
                    st.image((wm_atk * 255).astype(np.uint8), clamp=True, width=256)
                with col6:
                    # BER if a reference was provided earlier in Extract tab
                    if st.session_state["ref_wm_bits"] is not None:
                        ber_val, stats = bit_error_rate(st.session_state["ref_wm_bits"], bits_atk, return_counts=True)
                        st.metric("BER (JPEG)", f"{ber_val:.4f}")
                        st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")
                    elif st.session_state["last_wm_bits"] is not None:
                        # fallback to BER vs last embedded bits (if same watermark)
                        ber_val, stats = bit_error_rate(st.session_state["last_wm_bits"], bits_atk, return_counts=True)
                        st.metric("BER (JPEG vs last embedded)", f"{ber_val:.4f}")
                    else:
                        st.info("To show BER here, run Extract with an original watermark first.")

        # Small Cutout Attack
        with sub[1]:
            area = st.slider("Patch area ratio", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f", key="arear")
            nump = st.slider("Number of patches", 1, 150, 50, key="nump")
            shape = st.selectbox("Shape", ["rect", "circle"], index=0, key="shape")
            fill = st.selectbox("Fill", ["noise", "black", "avg", "blur"], index=0, key="fill")
            blur_k = st.slider("Blur kernel (odd)", 3, 31, 11, step=2, key="blurk")

            if st.button("Run Small Cutout Attack"):
                atk_rgb = small_random_cutout_mem(
                    in_rgb=wm_rgb, area_ratio=float(area), num_patches=int(nump),
                    shape=shape, fill=fill, blur_kernel=int(blur_k), seed=int(secret_key)
                )
                atk_pil = np_rgb_to_pil(atk_rgb)
                st.image(atk_pil, caption=f"Small cutouts ({nump} patches, {area:.4f} area each)", use_column_width=True)

                # Extract after attack
                y_atk, _, _ = rgb_to_ycbcr_arrays(atk_pil)
                bits_atk = extract_bits_y_multilevel_svd_qim(
                    y_luma=y_atk,
                    n_bits=WATERMARK_N * WATERMARK_N,
                    secret_key=int(secret_key),
                    quant_step=float(quant_step),
                    svd_index=int(svd_index),
                    svd_patch=((r0, r1), (c0, c1)),
                    wavelet=wavelet,
                    dwt_levels=int(dwt_levels),
                    tiles=(int(tiles_r), int(tiles_c)),
                    include_LL=bool(include_LL),
                    include_D=bool(include_D),
                )
                wm_atk = np.array(bits_atk, dtype=np.uint8).reshape(WATERMARK_N, WATERMARK_N)

                col7, col8 = st.columns(2)
                with col7:
                    st.markdown("**Extracted (Small Cutouts)**")
                    st.image((wm_atk * 255).astype(np.uint8), clamp=True, width=256)
                with col8:
                    if st.session_state["ref_wm_bits"] is not None:
                        ber_val, stats = bit_error_rate(st.session_state["ref_wm_bits"], bits_atk, return_counts=True)
                        st.metric("BER (Cutouts)", f"{ber_val:.4f}")
                        st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")
                    elif st.session_state["last_wm_bits"] is not None:
                        ber_val, stats = bit_error_rate(st.session_state["last_wm_bits"], bits_atk, return_counts=True)
                        st.metric("BER (Cutouts vs last embedded)", f"{ber_val:.4f}")
                    else:
                        st.info("To show BER here, run Extract with an original watermark first.")

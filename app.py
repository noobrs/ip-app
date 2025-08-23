import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import cv2
import pywt

# ---- import your core functions ----
from svd_qim_core import (
    psnr, bit_error_rate,
    embed_bits_y_multilevel_svd_qim, extract_bits_y_multilevel_svd_qim,
    save_small_random_cutout,  # we'll re-implement a memory variant below
)

# -------------------------------
# Helpers (memory-based I/O)
# -------------------------------
def pil_to_rgb_np(im: Image.Image) -> np.ndarray:
    """PIL RGB -> np.uint8 [H,W,3]"""
    return np.array(im.convert("RGB"), dtype=np.uint8)

def rgb_np_to_pil(arr: np.ndarray) -> Image.Image:
    """np.uint8 [H,W,3] -> PIL RGB"""
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")

def rgb_to_ycbcr_arrays(im_rgb: Image.Image):
    """Return (Y float64, Cb u8, Cr u8) from PIL RGB."""
    ycbcr = im_rgb.convert("YCbCr")
    y = np.array(ycbcr.getchannel(0), dtype=np.float64)
    cb = np.array(ycbcr.getchannel(1), dtype=np.uint8)
    cr = np.array(ycbcr.getchannel(2), dtype=np.uint8)
    return y, cb, cr

def ycbcr_arrays_to_rgb(y_luma: np.ndarray, cb_ch: np.ndarray, cr_ch: np.ndarray) -> Image.Image:
    y_u8 = np.rint(np.clip(y_luma, 0, 255)).astype(np.uint8)
    ycbcr_img = Image.merge(
        "YCbCr",
        (Image.fromarray(y_u8), Image.fromarray(cb_ch), Image.fromarray(cr_ch))
    )
    return ycbcr_img.convert("RGB")

def default_host(H=720, W=960):
    """Fallback colorful gradient host if user didn't upload."""
    x = np.linspace(0, 1, W, dtype=np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    r = (xv * 255).astype(np.uint8)
    g = (yv * 255).astype(np.uint8)
    b = ((1 - 0.5*xv - 0.5*yv) * 255).clip(0, 255).astype(np.uint8)
    return rgb_np_to_pil(np.dstack([r, g, b]))

def make_binary_watermark_from_pil(wm_pil: Image.Image, size=32):
    wm_gray = wm_pil.convert("L").resize((size, size), Image.LANCZOS)
    wm_bin = (np.array(wm_gray, dtype=np.uint8) > 127).astype(np.uint8)
    return wm_bin  # [H,W] of {0,1}

def random_watermark(size=32, seed=1234):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size)) > 0.5).astype(np.uint8)

def psnr_rgb(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """PSNR on 3-channel images via your psnr() (expects uint8, same shape)."""
    return psnr(rgb_a, rgb_b)

def numpy_image_download_button(img_pil: Image.Image, label: str, filename: str):
    buf = io.BytesIO()
    img_pil.save(buf, format=filename.split(".")[-1].upper())
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/" + filename.split(".")[-1])

# In-memory small cutout (adapted from your disk version)
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

    for _ in range(max(1, int(num_patches))):
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

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="SVD–QIM Watermarking Demo", layout="wide")
st.title("🔐 SVD–QIM Watermarking (DWT–DCT–SVD, Σ-only QIM)")

with st.sidebar:
    st.subheader("1) Inputs")
    host_file = st.file_uploader("Host image (RGB)", type=["png","jpg","jpeg","webp"])
    wm_file   = st.file_uploader("Watermark image (grayscale or RGB)", type=["png","jpg","jpeg","webp"])
    watermark_size = st.slider("Watermark size (NxN)", 16, 128, 32, step=8)

    st.subheader("2) Core Params")
    secret_key  = st.number_input("Secret key (int)", value=1234567890, step=1)
    quant_step  = st.slider("QIM quantization step (σ_k)", 10.0, 120.0, 80.0, step=1.0)
    wavelet     = st.selectbox("Wavelet", ["haar","db2","db4","sym4","coif1"], index=0)
    dwt_levels  = st.slider("DWT levels", 1, 4, 4, step=1)
    tiles_r = st.selectbox("Tiles (rows)", [1,2,3,4], index=1)
    tiles_c = st.selectbox("Tiles (cols)", [1,2,3,4], index=1)
    include_LL = st.checkbox("Include LL band", value=False)
    include_D  = st.checkbox("Include Diagonal band (D)", value=False)

    st.subheader("3) SVD patch")
    svd_index = st.selectbox("Singular value index k (0 = largest)", [0,1,2,3], index=0)
    r0 = st.slider("Patch row start", 0, 7, 2); r1 = st.slider("Patch row end (exclusive)", r0+1, 8, 6)
    c0 = st.slider("Patch col start", 0, 7, 2); c1 = st.slider("Patch col end (exclusive)", c0+1, 8, 6)

    st.caption("Tip: mid-band patch like rows/cols 2..6 works well for JPEG robustness.")

# Load or create host
if host_file:
    host_pil = Image.open(host_file).convert("RGB")
else:
    host_pil = default_host()
    st.info("No host uploaded — using a colorful synthetic image as demo.")

# Load or create watermark
if wm_file:
    wm_pil = Image.open(wm_file)
    wm_bin = make_binary_watermark_from_pil(wm_pil, size=watermark_size)
else:
    wm_bin = random_watermark(size=watermark_size, seed=secret_key)
    st.info("No watermark uploaded — using a deterministic random 0/1 pattern.")

wm_bits = wm_bin.reshape(-1).tolist()

# Show inputs
col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown("**Host**")
    st.image(host_pil, use_column_width=True)
with col_b:
    st.markdown("**Watermark (binary preview)**")
    st.image((wm_bin*255).astype(np.uint8), clamp=True, width=256)

# Session states
if "watermarked_rgb" not in st.session_state:
    st.session_state["watermarked_rgb"] = None
if "ycbcr_host" not in st.session_state:
    y, cb, cr = rgb_to_ycbcr_arrays(host_pil)
    st.session_state["ycbcr_host"] = (y, cb, cr)
if "wm_bits" not in st.session_state:
    st.session_state["wm_bits"] = wm_bits

# -------------------------------
# Embed
# -------------------------------
st.markdown("---")
st.header("Embed")
embed_clicked = st.button("🔧 Embed Watermark")

if embed_clicked:
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
    wm_rgb = ycbcr_arrays_to_rgb(y_wm, cb_host, cr_host)
    st.session_state["watermarked_rgb"] = pil_to_rgb_np(wm_rgb)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Watermarked image**")
        st.image(wm_rgb, use_column_width=True)
        numpy_image_download_button(wm_rgb, "⬇️ Download watermarked", "watermarked.png")

    with col2:
        host_rgb = pil_to_rgb_np(host_pil)
        psnr_val = psnr_rgb(host_rgb, st.session_state["watermarked_rgb"])
        st.metric("PSNR (Host vs Watermarked)", f"{psnr_val:.2f} dB")

# -------------------------------
# Extract (from any image)
# -------------------------------
st.markdown("---")
st.header("Extract")

extract_src = st.radio("Source", ["Current watermarked (above)", "Upload attacked / any image"], index=0)
uploaded_attack = None
if extract_src == "Upload attacked / any image":
    uploaded_attack = st.file_uploader("Attacked/Unknown image", type=["png","jpg","jpeg","webp"], key="attack_upl")

if st.button("🔎 Extract Watermark"):
    if extract_src == "Current watermarked (above)":
        if st.session_state["watermarked_rgb"] is None:
            st.warning("Please embed first or upload an attacked image.")
        else:
            src_pil = rgb_np_to_pil(st.session_state["watermarked_rgb"])
    else:
        if uploaded_attack is None:
            st.warning("Please upload an image to extract from.")
            st.stop()
        src_pil = Image.open(uploaded_attack).convert("RGB")

    y_src, _, _, = rgb_to_ycbcr_arrays(src_pil)
    bits_out = extract_bits_y_multilevel_svd_qim(
        y_luma=y_src,
        n_bits=watermark_size * watermark_size,
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
    wm_rec = np.array(bits_out, dtype=np.uint8).reshape(watermark_size, watermark_size)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Extracted watermark**")
        st.image((wm_rec * 255).astype(np.uint8), clamp=True, width=256)
    with col4:
        ber, stats = bit_error_rate(
            (st.session_state.get("wm_bits", wm_bits)),
            bits_out,
            return_counts=True
        )
        st.metric("BER", f"{ber:.4f}")
        st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")

# -------------------------------
# Built-in Attacks
# -------------------------------
st.markdown("---")
st.header("Attacks")

if st.session_state["watermarked_rgb"] is None:
    st.info("Embed first to run built-in attacks below.")
else:
    tabs = st.tabs(["JPEG Compression", "Small Cutouts"])
    with tabs[0]:
        q = st.slider("JPEG quality", 10, 95, 75, step=1)
        if st.button("Run JPEG Attack"):
            # Pillow re-encode to JPEG (memory)
            buf = io.BytesIO()
            rgb_np_to_pil(st.session_state["watermarked_rgb"]).save(buf, format="JPEG", quality=q, subsampling=0, optimize=False)
            buf.seek(0)
            atk_pil = Image.open(buf).convert("RGB")

            st.image(atk_pil, caption=f"JPEG q={q}", use_column_width=True)

            # Extract
            y_atk, _, _ = rgb_to_ycbcr_arrays(atk_pil)
            bits_atk = extract_bits_y_multilevel_svd_qim(
                y_luma=y_atk,
                n_bits=watermark_size * watermark_size,
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
            wm_atk = np.array(bits_atk, dtype=np.uint8).reshape(watermark_size, watermark_size)
            col5, col6 = st.columns(2)
            with col5:
                st.markdown("**Extracted (JPEG)**")
                st.image((wm_atk * 255).astype(np.uint8), clamp=True, width=256)
            with col6:
                ber, stats = bit_error_rate(st.session_state["wm_bits"], bits_atk, return_counts=True)
                st.metric("BER (JPEG)", f"{ber:.4f}")
                st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")

    with tabs[1]:
        area = st.slider("Patch area ratio", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
        nump = st.slider("Number of patches", 1, 150, 50)
        shape = st.selectbox("Shape", ["rect","circle"])
        fill = st.selectbox("Fill", ["noise","black","avg","blur"], index=0)
        blur_k = st.slider("Blur kernel (odd)", 3, 31, 11, step=2)
        if st.button("Run Small Cutout Attack"):
            atk_rgb = small_random_cutout_mem(
                in_rgb=st.session_state["watermarked_rgb"],
                area_ratio=float(area),
                num_patches=int(nump),
                shape=shape,
                fill=fill,
                blur_kernel=int(blur_k),
                seed=int(secret_key)
            )
            atk_pil = rgb_np_to_pil(atk_rgb)
            st.image(atk_pil, caption=f"Small cutouts ({nump} patches, {area:.4f} area each)", use_column_width=True)

            # Extract
            y_atk, _, _ = rgb_to_ycbcr_arrays(atk_pil)
            bits_atk = extract_bits_y_multilevel_svd_qim(
                y_luma=y_atk,
                n_bits=watermark_size * watermark_size,
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
            wm_atk = np.array(bits_atk, dtype=np.uint8).reshape(watermark_size, watermark_size)
            col7, col8 = st.columns(2)
            with col7:
                st.markdown("**Extracted (Small Cutouts)**")
                st.image((wm_atk * 255).astype(np.uint8), clamp=True, width=256)
            with col8:
                ber, stats = bit_error_rate(st.session_state["wm_bits"], bits_atk, return_counts=True)
                st.metric("BER (Cutouts)", f"{ber:.4f}")
                st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")

st.markdown("---")
with st.expander("About this prototype"):
    st.write("""
This Streamlit UI wraps an SVD–QIM watermarking pipeline:
- DWT→DCT 8×8→mid-band SVD (Σ-only) with QIM on σₖ
- Randomized block assignment, tiling, soft voting extraction
- Designed for robustness to JPEG compression and small local cutouts
    """)

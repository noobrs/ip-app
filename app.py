# app.py
import io
import zipfile
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
import cv2
import pywt
import graphviz

# ---- bring in your core functions (unchanged) ----
from svd_qim_core import (
    psnr, bit_error_rate,
    embed_bits_y_multilevel_svd_qim, extract_bits_y_multilevel_svd_qim,
    embed_bits_y_multilevel_svd_qim_batch, extract_bits_y_multilevel_svd_qim_batch,  # <-- NEW
    bands_all_levels,
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
    img.save(buf, format=fmt)
    return buf.getvalue()

def bytes_to_pil(b: bytes) -> Image.Image:
    """Convert raw bytes back to PIL RGB image."""
    return Image.open(io.BytesIO(b)).convert("RGB")

def np_image_download_button(img_pil: Image.Image, label: str, filename: str, unique_key: str):
    """Your safe download button for PIL images."""
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

# --- JPEG 2000 (JP2) attack in memory ---
def jpeg2000_attack_mem(in_rgb: np.ndarray, compression_x1000: int = 2000):
    """
    Encode -> decode as JPEG 2000 in memory.
    compression_x1000 ≈ compression ratio × 1000 (OpenCV flag).
    - 1000 ~ lossless-ish / low compression
    - larger values ~ stronger compression
    """
    bgr = cv2.cvtColor(in_rgb, cv2.COLOR_RGB2BGR)
    buf = None
    try:
        # Try explicit compression setting (if flag is present in this OpenCV build)
        buf_ok, buf = cv2.imencode(
            ".jp2", bgr,
            [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(compression_x1000)]
        )
        if not buf_ok:
            raise RuntimeError("OpenCV JP2 encode failed with COMPRESSION_X1000")
    except Exception:
        # Fallback: default JP2 encode (builds without the flag)
        buf_ok, buf = cv2.imencode(".jp2", bgr)
        if not buf_ok:
            raise RuntimeError("This OpenCV build lacks JPEG 2000 (.jp2) support.")

    # Decode back to RGB for downstream extraction
    arr = np.frombuffer(buf.tobytes(), dtype=np.uint8)
    bgr2 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr2 is None:
        raise RuntimeError("OpenCV JP2 decode failed.")
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB), buf.tobytes()  # (attacked_rgb, jp2_bytes)


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

tabs = st.tabs(["Embed", "Extract", "Attack", "Flow"])

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
    host_files = st.file_uploader(
        "Host image(s) (RGB). If none, a demo gradient will be used.",
        type=["png","jpg","jpeg","webp"],
        key="host_upl",
        accept_multiple_files=True
    )
    wm_file   = st.file_uploader(
        "Watermark image (will be binarized to fixed 32×32)",
        type=["png","jpg","jpeg","webp"],
        key="wm_upl"
    )

    # Prepare list of host PILs
    host_pils = []
    host_names = []
    if host_files:
        for f in host_files:
            host_pils.append(Image.open(f).convert("RGB"))
            host_names.append(getattr(f, "name", "host.png"))
    else:
        host_pils = [default_host()]
        host_names = ["demo.png"]
        st.info("No host uploaded — using a colorful synthetic image (1 item).")

    # Recommended minimum size (shown once; capacity per file is shown during loop)
    n_bits = WATERMARK_N * WATERMARK_N
    recN, capN = recommend_min_square_size(
        n_bits=n_bits,
        wavelet=wavelet,
        dwt_levels=int(dwt_levels),
        tiles=(int(tiles_r), int(tiles_c)),
        include_LL=bool(include_LL),
        include_D=bool(include_D),
    )
    st.markdown(f"**Recommended minimum host size (square)** for these settings: **{recN}×{recN}** "
                f"(capacity≈{capN} blocks for {n_bits} bits).")

    # Watermark bits (required)
    if wm_file:
        wm_pil = Image.open(wm_file)
        wm_bin = make_binary_watermark_from_pil(wm_pil, size=WATERMARK_N)
        wm_bits = wm_bin.reshape(-1).tolist()
    else:
        wm_bin = None
        wm_bits = None
        st.info("Please upload a watermark image (any size); it will be binarized to 32×32.")

    # Preview first host + watermark
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.markdown("**Sample Host (first)**")
        st.image(host_pils[0], use_column_width=True)
    with col_in2:
        st.markdown("**Watermark (fixed 32×32, binary preview)**")
        if wm_bin is not None:
            st.image((wm_bin * 255).astype(np.uint8), clamp=True, width=256)
        else:
            st.write("—")

    if st.button("🔧 Embed Watermark (Batch)"):
        if wm_bits is None:
            st.warning("Upload a watermark first.")
        else:
            # Convert all hosts to YCbCr arrays
            y_list, cb_list, cr_list = [], [], []
            for hp in host_pils:
                y, cb, cr = rgb_to_ycbcr_arrays(hp)
                y_list.append(y); cb_list.append(cb); cr_list.append(cr)

            # Batch-embed on Y
            y_wm_list = embed_bits_y_multilevel_svd_qim_batch(
                y_lumas=y_list,
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

            # Recombine and display per-file results
            st.session_state["last_wm_bits"] = wm_bits
            st.session_state["watermarked_rgb"] = None  # will set for the last item (keeps Attack tab behavior)

            for idx, (hp, y_wm, cb, cr, name) in enumerate(zip(host_pils, y_wm_list, cb_list, cr_list, host_names), start=1):
                wm_rgb_pil = ycbcr_arrays_to_rgb(y_wm, cb, cr)
                wm_rgb = pil_to_np_rgb(wm_rgb_pil)
                host_rgb = pil_to_np_rgb(hp)

                # Save last item for Attack tab
                st.session_state["watermarked_rgb"] = wm_rgb

                # Capacity for this file
                H, W = hp.size[1], hp.size[0]
                file_capacity = total_capacity_for_shape(
                    H, W, wavelet, int(dwt_levels), (int(tiles_r), int(tiles_c)), bool(include_LL), bool(include_D)
                )

                st.markdown(f"### Result #{idx}: `{name}`  ·  Image size: **{W}×{H}**  ·  Capacity≈**{file_capacity}** blocks")
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.markdown("**Original**")
                    st.image(get_preview_image(hp))
                with col_b:
                    st.markdown("**Watermarked**")
                    st.image(get_preview_image(wm_rgb_pil))

                psnr_val = utilities.psnr(utilities.pil_to_np_rgb(hp), utilities.pil_to_np_rgb(wm_rgb_pil))
                st.metric("PSNR (Original vs Watermarked)", f"{psnr_val:.2f} dB")
                st.download_button(
                    f"Download `{name}` (watermarked)",
                    data=out_pairs[-1][1],
                    file_name=out_name,
                    mime="image/png",
                    use_container_width=False,
                    key=f"dl_wm_{name}"
                )
                st.markdown("---")

# ==========================================================
# TAB 2: EXTRACT
# ==========================================================
with tabs[1]:
    st.header("Extract")
    st.caption("Upload only the watermarked image for extraction. "
               "Upload the original watermark (32×32) if you want BER.")

    attacked_files = st.file_uploader(
        "Watermarked image(s) to extract from",
        type=["png","jpg","jpeg","webp"],
        key="attacked_upl_extract",
        accept_multiple_files=True
    )
    ori_wm_file   = st.file_uploader(
        "Original watermark image (for BER, optional; will be binarized to 32×32)",
        type=["png","jpg","jpeg","webp"],
        key="oriwm_upl_extract"
    )

    if st.button("🔎 Extract Watermark(s)"):
        if not attacked_files:
            st.warning("Please upload at least one watermarked image.")
        else:
            # Reference watermark bits (optional)
            ref_bits = None
            if ori_wm_file:
                ori_wm_pil = Image.open(ori_wm_file)
                ori_wm_bin = make_binary_watermark_from_pil(ori_wm_pil, size=WATERMARK_N)
                ref_bits = ori_wm_bin.reshape(-1).tolist()
                st.session_state["ref_wm_bits"] = ref_bits

            # Build lists of Y from all attacked files
            atk_pils, atk_names, y_list = [], [], []
            for f in attacked_files:
                pil_img = Image.open(f).convert("RGB")
                atk_pils.append(pil_img)
                atk_names.append(getattr(f, "name", "watermarked.png"))
                y, _, _ = rgb_to_ycbcr_arrays(pil_img)
                y_list.append(y)

            # Batch extract bits
            bits_list = extract_bits_y_multilevel_svd_qim_batch(
                y_lumas=y_list,
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

            # Show per-file results
            for idx, (name, bits_out) in enumerate(zip(atk_names, bits_list), start=1):
                wm_rec = np.array(bits_out, dtype=np.uint8).reshape(WATERMARK_N, WATERMARK_N)
                st.markdown(f"### Extracted #{idx}: `{name}`")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Source image**")
                    st.image(atk_pils[idx-1], use_column_width=True)
                with col2:
                    st.markdown("**Recovered 32×32 watermark**")
                    st.image((wm_rec * 255).astype(np.uint8), clamp=True, width=256)

                if ref_bits is not None:
                    ber_val, stats = bit_error_rate(ref_bits, bits_out, return_counts=True)
                    st.metric("BER", f"{ber_val:.4f}")
                    st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")
                else:
                    st.info("Upload the original 32×32 watermark image above to compute BER.")
                st.markdown("---")


# ----------------------------
# TAB 3: ATTACK
# ----------------------------
def tab_attack():
    st.header("Attack Simulation")
    st.info("Upload watermarked images to simulate attacks and download the attacked versions.")

    attack_tabs = st.tabs(["JPEG Compression", "Cropping", "Small Rotation"])

    # --- JPEG ---
    with attack_tabs[0]:
        st.markdown("### JPEG Attack")
        jpeg_files = st.file_uploader(
            "Upload one or more original watermarked images",
            type=["png", "jpg", "jpeg", "webp"],
            key="jpeg_attack_files",
            accept_multiple_files=True,
        )

        if not jpeg_files:
            st.info("Upload images to see JPEG attack preview.")
        else:
            quality = st.slider("JPEG Quality", 10, 95, 75, step=1, key="jpeg_quality_slider")
            out_list = []
            for f in jpeg_files:
                try:
                    p = Image.open(f).convert("RGB")
                    if not aspect_ok(p):
                        st.error(f"Rejected `{getattr(f,'name','file')}`: aspect ratio too extreme.")
                        continue
                    p, _ = resize_into_range(p, 512, 1080)

                    # compress → bytes
                    buf = io.BytesIO()
                    p.save(buf, format="JPEG", quality=quality, subsampling=0, optimize=False)
                    jpeg_bytes = buf.getvalue()

                    # preview (exact file to be downloaded)
                    prev = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                    name = getattr(f, "name", "original.png")
                    col1, col2 = st.columns(2, gap="large")
                    with col1:
                        st.markdown("**Original**")
                        st.image(get_preview_image(p), caption=f"{p.width}×{p.height}px")
                    with col2:
                        st.markdown(f"**JPEG Q{quality}**")
                        st.image(get_preview_image(prev), caption=f"{prev.width}×{prev.height}px")

                    out_name = f"{name.rsplit('.',1)[0]}_jpeg_q{quality}.jpg"
                    st.download_button(
                        f"Download JPEG Q{quality}",
                        data=jpeg_bytes,
                        file_name=out_name,
                        mime="image/jpeg",
                        use_container_width=False,
                        key=f"jpeg_download_{name}_{quality}"
                    )
                    out_list.append((out_name, jpeg_bytes))
                    st.markdown("---")
                except Exception as e:
                    st.error(f"JPEG 2000 not available in this environment: {e}")
                    st.stop()

                atk_pil = np_rgb_to_pil(atk_rgb)
                st.image(atk_pil, caption=f"JPEG 2000 (×1000={comp})", use_column_width=True)

                # Optional: download the attacked JP2 file
                st.download_button(
                    f"Download ALL JPEG Q{quality} (ZIP)",
                    data=zip_buffer,
                    file_name=f"jpeg_attacked_batch_q{quality}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_zip_jpeg"
                )

    # --- Cropping ---
    with attack_tabs[1]:
        st.markdown("### Cropping Attack")
        cropping_files = st.file_uploader(
            "Upload one or more original watermarked images",
            type=["png", "jpg", "jpeg", "webp"],
            key="cropping_attack_files",
            accept_multiple_files=True,
        )

        if not cropping_files:
            st.info("Upload images to see cropping attack preview.")
        else:
            area_ratio = st.slider("Area Ratio", 0.01, 0.1, 0.05, step=0.01, format="%.2f", key="crop_area_ratio")
            num_patches = st.slider("Number of Patches", 1, 10, 1, key="crop_num_patches")
            seed = st.slider("Seed", 0, 100, 42, key="crop_seed")

            out_list = []
            for f in cropping_files:
                try:
                    p = Image.open(f).convert("RGB")
                    if not aspect_ok(p):
                        st.error(f"Rejected `{getattr(f,'name','file')}`: aspect ratio too extreme.")
                        continue
                    p, _ = resize_into_range(p, 512, 1080)
                    attacked = attacks.crop_attack(
                        in_arr=np.array(p),
                        area_ratio=area_ratio,
                        num_patches=num_patches,
                        seed=seed
                    )
                    attacked_pil = attacked if isinstance(attacked, Image.Image) else Image.fromarray(attacked.astype(np.uint8))
                    name = getattr(f, "name", "original.png")

                    col1, col2 = st.columns(2, gap="large")
                    with col1:
                        st.markdown("**Original**")
                        st.image(get_preview_image(p), caption=f"{p.width}×{p.height}px")
                    with col2:
                        st.markdown(f"**Cropped (Area: {area_ratio:.4f}, Patches: {num_patches}, Seed: {seed})**")
                        st.image(get_preview_image(attacked_pil), caption=f"{attacked_pil.width}×{attacked_pil.height}px")

                    out_name = f"{name.rsplit('.',1)[0]}_cropped_a{area_ratio}_p{num_patches}_s{seed}.png"
                    np_image_download_button(
                        attacked_pil,
                        "Download Cropped",
                        out_name,
                        unique_key=f"cropping_download_{name}_{area_ratio}_{num_patches}_{seed}"
                    )
                    out_list.append((out_name, pil_to_bytes(attacked_pil, "PNG")))
                    st.markdown("---")
                except Exception as e:
                    st.warning(f"Failed: {e}")

            st.session_state.attack["crop"] = out_list

            if len(out_list) > 1:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, b in out_list: zf.writestr(fname, b)
                zip_buffer.seek(0)
                st.download_button(
                    "Download ALL Cropped images (ZIP)",
                    data=zip_buffer,
                    file_name=f"cropped_attacked_batch_a{area_ratio}_p{num_patches}_s{seed}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_zip_crop"
                )

    # --- Small Rotation ---
    with attack_tabs[2]:
        st.markdown("### Small Rotation Attack")
        rotation_files = st.file_uploader(
            "Upload one or more original watermarked images",
            type=["png", "jpg", "jpeg", "webp"],
            key="rotation_attack_files",
            accept_multiple_files=True,
        )

        if not rotation_files:
            st.info("Upload images to see rotation attack preview.")
        else:
            angle = st.slider("Rotation Angle (degrees)", -5.0, 5.0, 1.0, step=0.5, key="rotation_angle")
            fill = (0, 0, 0)  # fixed black fill

            out_list = []
            for f in rotation_files:
                try:
                    p = Image.open(f).convert("RGB")
                    if not aspect_ok(p):
                        st.error(f"Rejected `{getattr(f,'name','file')}`: aspect ratio too extreme.")
                        continue
                    p, _ = resize_into_range(p, 512, 1080)
                    rotated = attacks.rotation_attack(p, angle=angle, fill_color=fill)
                    name = getattr(f, "name", "original.png")

                    col1, col2 = st.columns(2, gap="large")
                    with col1:
                        st.markdown("**Original**")
                        st.image(get_preview_image(p), caption=f"{p.width}×{p.height}px")
                    with col2:
                        st.markdown(f"**Rotated ({angle}°)**")
                        st.image(get_preview_image(rotated), caption=f"{rotated.width}×{rotated.height}px")

                    out_name = f"{name.rsplit('.',1)[0]}_rotated_{angle}deg.png"
                    np_image_download_button(
                        rotated,
                        "Download Rotated",
                        out_name,
                        unique_key=f"rotation_download_{name}_{angle}"
                    )
                    out_list.append((out_name, pil_to_bytes(rotated, "PNG")))
                    st.markdown("---")
                except Exception as e:
                    st.warning(f"Failed: {e}")

            st.session_state.attack["rotate"] = out_list

            if len(out_list) > 1:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, b in out_list: zf.writestr(fname, b)
                zip_buffer.seek(0)
                st.download_button(
                    f"Download ALL Rotated images (ZIP)",
                    data=zip_buffer,
                    file_name=f"rotated_attacked_batch_{angle}deg_black.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_zip_rot"
                )

# ----------------------------
# Main tabs
# ----------------------------
tabs = st.tabs(["Embed", "Extract", "Attack"])
with tabs[0]: tab_embed()
with tabs[1]: tab_extract()
with tabs[2]: tab_attack()
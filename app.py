# app.py
import io
import zipfile
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Your project modules
import watermarking, utilities, attacks, parameters

# ----------------------------
# Page setup & light styling
# ----------------------------
st.set_page_config(page_title="Image Watermarking", layout="wide")
st.markdown("""
<style>
/* Hide the built-in download link inside st_canvas only */
div[class*="st-drawable-canvas"] a[download] { display: none !important; }

/* Make section titles a bit tighter */
.block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; }

/* (Optional) Slightly bolder radios */
.st-emotion-cache-1q7i0r7 { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("Image Watermarking")

# ----------------------------
# Constants & small utilities
# ----------------------------
PREVIEW_SIZE = (600, 600)

def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    """Convert PIL image to raw bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def bytes_to_pil(b: bytes) -> Image.Image:
    """Convert raw bytes back to PIL RGB image."""
    return Image.open(io.BytesIO(b)).convert("RGB")

def np_image_download_button(img_pil: Image.Image, label: str, filename: str, unique_key: str):
    """Your safe download button for PIL images."""
    fmt = filename.split(".")[-1].upper()
    data = pil_to_bytes(img_pil, fmt="PNG" if fmt == "PNG" else fmt)
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="image/" + filename.split(".")[-1],
        use_container_width=False,
        key=unique_key,
    )

def get_preview_image(pil_img: Image.Image) -> Image.Image:
    """Create a fixed-size preview image while maintaining aspect ratio."""
    img_ratio = pil_img.width / pil_img.height
    preview_ratio = PREVIEW_SIZE[0] / PREVIEW_SIZE[1]
    if img_ratio > preview_ratio:
        new_w = PREVIEW_SIZE[0]
        new_h = int(PREVIEW_SIZE[0] / img_ratio)
    else:
        new_h = PREVIEW_SIZE[1]
        new_w = int(PREVIEW_SIZE[1] * img_ratio)
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def resize_into_range(p: Image.Image, lo=512, hi=1080) -> Tuple[Image.Image, bool]:
    """Keep aspect ratio; scale so both sides lie within [lo, hi]."""
    needs_resize = False
    W, H = p.width, p.height
    if W < lo or H < lo or W > hi or H > hi:
        needs_resize = True
        if W < lo or H < lo:
            s = max(lo / W, lo / H)
        else:
            s = min(hi / W, hi / H)
        new_w = int(W * s)
        new_h = int(H * s)
        new_w = max(lo, min(hi, new_w))
        new_h = max(lo, min(hi, new_h))
        p = p.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return p, needs_resize

def aspect_ok(p: Image.Image, max_ratio=3.0) -> bool:
    """Reject extremely elongated images (too unbalanced)."""
    r = max(p.width, p.height) / min(p.width, p.height)
    return r <= max_ratio

def show_grid_previews(pils: List[Image.Image], names: List[str], per_row=3):
    """Simple grid preview for a list of PIL images."""
    for i in range(0, len(pils), per_row):
        cols = st.columns(per_row, gap="medium")
        for j, col in enumerate(cols):
            k = i + j
            if k >= len(pils):
                break
            with col:
                prev = get_preview_image(pils[k])
                st.image(prev, caption=f"{names[k]} • {pils[k].width}×{pils[k].height}px")

# ----------------------------
# Session state bootstrapping
# ----------------------------
def init_state():
    ss = st.session_state
    # Everything related to the "Embed" tab
    if "embed" not in ss:
        ss.embed = {
            "host_bytes": [],      # list[bytes] of uploaded/processed hosts
            "host_names": [],      # list[str]
            "wm_bits": None,       # 1D list[int] or np.ndarray
            "wm_source": None,     # "upload" or "draw"
            "watermarked": [],     # list[(filename, bytes)]
        }
    # Everything related to the "Extract" tab
    if "extract" not in ss:
        ss.extract = {
            "atk_bytes": [],       # list[bytes] watermarked inputs
            "atk_names": [],
            "ref_bits": None,      # optional reference watermark bits
            "results": [],         # extraction outputs (list[(name, bits, preview_bytes)])
        }
    # Everything related to the "Attack" tab
    if "attack" not in ss:
        ss.attack = {
            "jpeg": [],            # list[(name, attacked_bytes)]
            "crop": [],
            "rotate": [],
        }

init_state()

# ----------------------------
# TAB 1: EMBED
# ----------------------------
def tab_embed():
    st.header("Embed Watermark")

    # --- HOST IMAGES ---
    st.markdown("### Host Image(s)")
    host_files = st.file_uploader(
        "Upload one or more host images.",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="host_upl_embed",   # STABLE KEY
    )

    # If user uploads new files, replace current host list with processed versions
    if host_files:
        host_pils, host_names = [], []
        for uf in host_files:
            try:
                pil = Image.open(uf).convert("RGB")
                if not aspect_ok(pil):
                    st.error(f"Rejected `{getattr(uf, 'name', 'file')}`: aspect ratio too extreme ({pil.width}×{pil.height}).")
                    continue
                original = (pil.width, pil.height)
                pil, resized = resize_into_range(pil, 512, 1080)
                if resized:
                    st.info(f"Auto-resized `{getattr(uf, 'name', 'file')}`: {original[0]}×{original[1]} → {pil.width}×{pil.height}")
                    host_names.append(f"{getattr(uf, 'name', 'unnamed')}_resized")
                else:
                    host_names.append(getattr(uf, "name", "unnamed"))
                host_pils.append(pil)
            except Exception as e:
                st.warning(f"Failed to open {getattr(uf,'name','file')}: {e}")

        # Persist to session_state as BYTES (robust across reruns)
        st.session_state.embed["host_bytes"] = [pil_to_bytes(p) for p in host_pils]
        st.session_state.embed["host_names"] = host_names

    # Show previews (from session_state)
    if len(st.session_state.embed["host_bytes"]) == 0:
        st.info("No image(s) yet — upload above to see previews.")
    else:
        st.markdown("**Image previews**")
        pils = [bytes_to_pil(b) for b in st.session_state.embed["host_bytes"]]
        show_grid_previews(pils, st.session_state.embed["host_names"])

    # --- WATERMARK SOURCE ---
    st.markdown("### Watermark Image")
    wm_source = st.radio(
        "Choose watermark source",
        ["Upload watermark", "Draw watermark"],
        horizontal=True,
        key="wm_source_embed"
    )

    wm_bits = None
    if wm_source == "Upload watermark":
        wm_file = st.file_uploader(
            "Upload watermark image (will be binarized to 32×32)",
            type=["png", "jpg", "jpeg", "webp"],
            key="wm_upl_embed"   # STABLE KEY
        )
        if wm_file is not None:
            wm_bits = utilities.prepare_watermark_bits(Image.open(wm_file), size=parameters.WATERMARK_SIZE)
    else:
        st.info("Draw in black on a white canvas. The drawing is binarized to 32×32.")
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=8,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=256, width=256,
            drawing_mode="freedraw",
            key="wm_canvas_embed",  # STABLE KEY
        )
        if canvas and canvas.image_data is not None:
            wm_bits = utilities.prepare_watermark_bits(Image.fromarray(canvas.image_data.astype(np.uint8)), size=parameters.WATERMARK_SIZE)

    # Persist watermark bits if present this run
    if wm_bits is not None:
        st.session_state.embed["wm_bits"] = wm_bits
        st.session_state.embed["wm_source"] = "upload" if (wm_source == "Upload watermark") else "draw"

    # --- WATERMARK PREVIEW ---
    if st.session_state.embed["wm_bits"] is not None:
        st.markdown("**Watermark preview**")
        wm_bin = np.array(st.session_state.embed["wm_bits"], dtype=np.uint8).reshape(parameters.WATERMARK_SIZE, parameters.WATERMARK_SIZE)
        big = Image.fromarray((wm_bin * 255).astype(np.uint8), mode="L").resize((256, 256), Image.NEAREST)
        st.image(big, clamp=True)

        # If drawn, allow download of the exact 32×32 binary
        if st.session_state.embed["wm_source"] == "draw":
            buf = io.BytesIO()
            Image.fromarray((wm_bin * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "Download watermark (32×32 PNG)",
                data=buf,
                file_name="watermark_32x32.png",
                mime="image/png",
                use_container_width=False,
                key="dl_drawn_wm_embed"
            )
    else:
        st.info("No watermark yet — upload or draw to see preview.")

        # --- EMBED ACTION ---
    if (len(st.session_state.embed["host_bytes"]) > 0) and (st.session_state.embed["wm_bits"] is not None):
        if st.button("Embed", key="embed_button"):
            hosts_b = st.session_state.embed["host_bytes"]
            wm_bits_s = st.session_state.embed["wm_bits"]

            out_pairs = []  # (filename, bytes)
            for name, hb in zip(st.session_state.embed["host_names"], hosts_b):
                hp = bytes_to_pil(hb)
                y, cb, cr = utilities.to_ycbcr_arrays(hp)
                y_wm = watermarking.embed_watermark(y, wm_bits_s)
                wm_rgb_pil = utilities.from_ycbcr_arrays(y_wm, cb, cr)
                out_name = f"{name.rsplit('.',1)[0]}_watermarked.png"
                out_pairs.append((out_name, pil_to_bytes(wm_rgb_pil, "PNG")))

                # Show side-by-side + PSNR + download
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.markdown("**Original**")
                    st.image(get_preview_image(hp))
                with col_b:
                    st.markdown("**Watermarked**")
                    st.image(get_preview_image(wm_rgb_pil))

                psnr_val = utilities.psnr(utilities.pil_to_np_rgb(hp), utilities.pil_to_np_rgb(wm_rgb_pil))
                st.metric("PSNR (Original vs Watermarked)", f"{psnr_val:.4f} dB")
                st.download_button(
                    f"Download watermarked image",
                    data=out_pairs[-1][1],
                    file_name=out_name,
                    mime="image/png",
                    use_container_width=False,
                    key=f"dl_wm_{name}"
                )
                st.markdown("---")

            # Persist batch for optional ZIP
            st.session_state.embed["watermarked"] = out_pairs

            if len(out_pairs) > 1:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, b in out_pairs:
                        zf.writestr(fname, b)
                zip_buffer.seek(0)
                st.download_button(
                    "Download ALL watermarked images (ZIP)",
                    data=zip_buffer,
                    file_name="watermarked_batch.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_zip_embed"
                )


# ----------------------------
# TAB 2: EXTRACT
# ----------------------------
def tab_extract():
    st.header("Extract Watermark")

    # --- Upload attacked/watermarked inputs ---
    st.markdown("### Watermarked Images")
    attacked_files = st.file_uploader(
        "Upload one or more watermarked images to extract watermarks.",
        type=["png","jpg","jpeg","webp"],
        key="attacked_upl_extract",
        accept_multiple_files=True
    )

    if attacked_files:
        atk_pils, atk_names = [], []
        for f in attacked_files:
            try:
                p = Image.open(f).convert("RGB")
                if not aspect_ok(p):
                    st.error(f"Rejected `{getattr(f,'name','file')}`: aspect ratio too extreme.")
                    continue
                orig = (p.width, p.height)
                p, resized = resize_into_range(p, 512, 1080)
                if resized:
                    st.info(f"Auto-resized `{getattr(f,'name','file')}`: {orig[0]}×{orig[1]} → {p.width}×{p.height}")
                    atk_names.append(f"{getattr(f, 'name', 'watermarked.png')}_resized")
                else:
                    atk_names.append(getattr(f, "name", "watermarked.png"))
                atk_pils.append(p)
            except Exception as e:
                st.warning(f"Failed to open {getattr(f,'name','file')}: {e}")

        st.session_state.extract["atk_bytes"] = [pil_to_bytes(p) for p in atk_pils]
        st.session_state.extract["atk_names"] = atk_names

    # Grid preview
    if len(st.session_state.extract["atk_bytes"]) == 0:
        st.info("Upload watermarked images above to extract.")
    else:
        st.markdown("**Watermarked image previews**")
        show_grid_previews(
            [bytes_to_pil(b) for b in st.session_state.extract["atk_bytes"]],
            st.session_state.extract["atk_names"]
        )

    # --- Optional reference watermark ---
    st.markdown("### Reference Watermark (Optional)")
    ori_wm_file = st.file_uploader(
        "Upload the original watermark to compute BER & NCC.",
        type=["png","jpg","jpeg","webp"],
        key="oriwm_upl_extract"
    )

    if ori_wm_file:
        try:
            st.markdown("**Reference watermark preview**")
            ori_pil = Image.open(ori_wm_file).convert("RGBA")
            bg = Image.new("RGBA", ori_pil.size, (255, 255, 255, 255))
            ori_rgb = Image.alpha_composite(bg, ori_pil).convert("RGB")
            ori_wm_bin = utilities.prepare_watermark_bits(ori_rgb, size=parameters.WATERMARK_SIZE)
            st.session_state.extract["ref_bits"] = ori_wm_bin.reshape(-1).tolist()

            big = Image.fromarray((ori_wm_bin * 255).astype(np.uint8), mode="L").resize((256, 256), Image.NEAREST)
            st.image(big, clamp=True)
        except Exception as e:
            st.warning(f"Failed to read original watermark: {e}")
            st.session_state.extract["ref_bits"] = None

    else:
        st.info("No watermark yet — upload or draw to see preview.")

    # --- EXTRACTION METHOD SELECTION ---
    st.markdown("### Extraction Method")
    extract_method = st.radio(
        "Choose extraction algorithm",
        ["Standard Algorithm", "Enhanced Algorithm (for rotation)", "Compare Both"],
        horizontal=True,
        key="extract_method_select"
    )
    

        # --- EXTRACT ACTION ---
    # Show the button only when at least one attacked/watermarked image is uploaded
    if len(st.session_state.extract["atk_bytes"]) > 0:
        if st.button("Extract", key="extract_button"):
            results = []
            for name, b in zip(st.session_state.extract["atk_names"], st.session_state.extract["atk_bytes"]):
                pil_img = bytes_to_pil(b)
                y, _, _ = utilities.to_ycbcr_arrays(pil_img)

                if extract_method == "Standard Algorithm":
                    bits = watermarking.extract_watermark(y)
                    wm_rec = np.array(bits, dtype=np.uint8).reshape(parameters.WATERMARK_SIZE, parameters.WATERMARK_SIZE)
                    rec_big = Image.fromarray((wm_rec * 255).astype(np.uint8), mode="L").resize((256, 256), Image.NEAREST)
                    results.append((name, bits, pil_to_bytes(rec_big, "PNG"), None, None))

                elif extract_method == "Enhanced Algorithm (for rotation)":
                    bits = watermarking.extract_watermark_enhanced(y)
                    wm_rec = np.array(bits, dtype=np.uint8).reshape(parameters.WATERMARK_SIZE, parameters.WATERMARK_SIZE)
                    rec_big = Image.fromarray((wm_rec * 255).astype(np.uint8), mode="L").resize((256, 256), Image.NEAREST)
                    results.append((name, bits, pil_to_bytes(rec_big, "PNG"), None, None))

                else:  # Compare Both
                    bits_std = watermarking.extract_watermark(y)
                    bits_enh = watermarking.extract_watermark_enhanced(y)

                    wm_rec_std = np.array(bits_std, dtype=np.uint8).reshape(parameters.WATERMARK_SIZE, parameters.WATERMARK_SIZE)
                    wm_rec_enh = np.array(bits_enh, dtype=np.uint8).reshape(parameters.WATERMARK_SIZE, parameters.WATERMARK_SIZE)

                    rec_big_std = Image.fromarray((wm_rec_std * 255).astype(np.uint8), mode="L").resize((256, 256), Image.NEAREST)
                    rec_big_enh = Image.fromarray((wm_rec_enh * 255).astype(np.uint8), mode="L").resize((256, 256), Image.NEAREST)

                    results.append((name, bits_std, pil_to_bytes(rec_big_std, "PNG"), bits_enh, pil_to_bytes(rec_big_enh, "PNG")))

            st.session_state.extract["results"] = results

            # Show per-file results + metrics if ref available
            for idx, result in enumerate(results, start=1):
                if extract_method == "Compare Both":
                    name, bits_std, preview_std_png, bits_enh, preview_enh_png = result
                    st.markdown(f"### Extracted #{idx}: `{name}`")

                    col1, col2, col3 = st.columns(3, gap="medium")
                    with col1:
                        st.markdown("**Source image**")
                        st.image(get_preview_image(bytes_to_pil(st.session_state.extract["atk_bytes"][idx-1])))
                    with col2:
                        st.markdown("**Standard Algorithm**")
                        st.image(Image.open(io.BytesIO(preview_std_png)), clamp=True)
                    with col3:
                        st.markdown("**Enhanced Algorithm**")
                        st.image(Image.open(io.BytesIO(preview_enh_png)), clamp=True)

                    ref_bits = st.session_state.extract["ref_bits"]
                    if ref_bits is not None:
                        ber_std, stats_std = utilities.bit_error_rate(ref_bits, bits_std, return_counts=True)
                        ncc_std = utilities.normalized_cross_correlation(ref_bits, bits_std)

                        ber_enh, stats_enh = utilities.bit_error_rate(ref_bits, bits_enh, return_counts=True)
                        ncc_enh = utilities.normalized_cross_correlation(ref_bits, bits_enh)

                        st.markdown("#### Metrics Comparison")
                        col_std, col_enh = st.columns(2, gap="large")
                        with col_std:
                            st.markdown("**Standard Algorithm**")
                            c1, c2 = st.columns(2)
                            with c1: st.metric("BER", f"{ber_std:.4f}")
                            with c2: st.metric("NCC", f"{ncc_std:.4f}")
                            st.write(f"Accuracy: **{stats_std['accuracy']:.4f}**  |  Errors: **{stats_std['errors']}/{stats_std['total']}**")
                        with col_enh:
                            st.markdown("**Enhanced Algorithm**")
                            c1, c2 = st.columns(2)
                            with c1: st.metric("BER", f"{ber_enh:.4f}")
                            with c2: st.metric("NCC", f"{ncc_enh:.4f}")
                            st.write(f"Accuracy: **{stats_enh['accuracy']:.4f}**  |  Errors: **{stats_enh['errors']}/{stats_enh['total']}**")

                        if stats_std['accuracy'] > stats_enh['accuracy']:
                            st.success("Standard algorithm performed better for this image")
                        elif stats_enh['accuracy'] > stats_std['accuracy']:
                            st.success("Enhanced algorithm performed better for this image")
                        else:
                            st.info("Both algorithms performed equally")
                    else:
                        st.info("Upload the reference watermark above to compute BER and NCC metrics.")

                else:
                    name, bits_out, preview_png = result[:3]
                    st.markdown(f"### Extracted #{idx}: `{name}`")
                    col1, col2 = st.columns(2, gap="large")
                    with col1:
                        st.markdown("**Source image**")
                        st.image(get_preview_image(bytes_to_pil(st.session_state.extract["atk_bytes"][idx-1])))
                    with col2:
                        method_name = "Standard" if extract_method == "Standard Algorithm" else "Enhanced"
                        st.markdown(f"**Recovered watermark ({method_name})**")
                        st.image(Image.open(io.BytesIO(preview_png)), clamp=True)

                    ref_bits = st.session_state.extract["ref_bits"]
                    if ref_bits is not None:
                        ber_val, stats = utilities.bit_error_rate(ref_bits, bits_out, return_counts=True)
                        ncc_val = utilities.normalized_cross_correlation(ref_bits, bits_out)
                        c1, c2 = st.columns(2)
                        with c1: st.metric("BER", f"{ber_val:.4f}")
                        with c2: st.metric("NCC", f"{ncc_val:.4f}")
                        st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")
                    else:
                        st.info("Upload the reference watermark above to compute BER and NCC metrics.")

                st.markdown("---")


# ----------------------------
# TAB 3: ATTACK
# ----------------------------
def tab_attack():
    st.header("Attack Simulation")
    st.info("Upload watermarked images to simulate attacks and download the attacked versions.")

    attack_tabs = st.tabs(["JPEG Compression", "Cropping", "Small Rotation", "Additive Noise"])

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
                    st.warning(f"Failed: {e}")

            st.session_state.attack["jpeg"] = out_list

            if len(out_list) > 1:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, b in out_list: zf.writestr(fname, b)
                zip_buffer.seek(0)
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
    
    # -- Additive Noise -- 
    with attack_tabs[3]:
        st.markdown("### Additive Noise Attack")
        noise_files = st.file_uploader(
            "Upload one or more original watermarked images",
            type=["png", "jpg", "jpeg", "webp"],
            key="noise_attack_files",
            accept_multiple_files=True,
        )

        if not noise_files:
            st.info("Upload images to see noise attack preview.")
        else:
            # Noise type selection - REMOVED Salt & Pepper
            noise_type = st.radio(
                "Select noise type:",
                ["Gaussian Noise", "Uniform Noise"],
                horizontal=True,
                key="noise_type_selection"
            )
            
            # Parameters based on noise type
            if noise_type == "Gaussian Noise":
                std = st.slider("Standard Deviation", 1, 15, 10, key="gaussian_std")
                param_text = f"σ={std}"
                param_short = f"std{std}"
            else:  # Uniform Noise
                intensity = st.slider("Noise Intensity", 1, 25, 20, key="uniform_intensity")
                param_text = f"intensity={intensity}"
                param_short = f"uni{intensity}"
            
            seed = st.slider("Seed", 0, 100, 42, key="noise_seed")

            out_list = []
            for f in noise_files:
                try:
                    p = Image.open(f).convert("RGB")
                    if not aspect_ok(p):
                        st.error(f"Rejected {getattr(f,'name','file')}: aspect ratio too extreme.")
                        continue
                    p, _ = resize_into_range(p, 512, 1080)
                    
                    # Apply noise attack based on selection - REMOVED Salt & Pepper
                    arr = np.array(p)
                    if noise_type == "Gaussian Noise":
                        attacked_arr = attacks.gaussian_noise_attack(arr, std=std, seed=seed)
                    else:  # Uniform Noise
                        attacked_arr = attacks.uniform_noise_attack(arr, intensity=intensity, seed=seed)
                    
                    attacked_pil = Image.fromarray(attacked_arr.astype(np.uint8))
                    name = getattr(f, "name", "original.png")

                    col1, col2 = st.columns(2, gap="large")
                    with col1:
                        st.markdown("*Original*")
                        st.image(get_preview_image(p), caption=f"{p.width}×{p.height}px")
                    with col2:
                        st.markdown(f"{noise_type} ({param_text})")
                        st.image(get_preview_image(attacked_pil), caption=f"{attacked_pil.width}×{attacked_pil.height}px")

                    out_name = f"{name.rsplit('.',1)[0]}_{param_short}_s{seed}.png"
                    np_image_download_button(
                        attacked_pil,
                        f"Download {noise_type}",
                        out_name,
                        unique_key=f"noise_download_{name}{param_short}{seed}"
                    )
                    out_list.append((out_name, pil_to_bytes(attacked_pil, "PNG")))
                    st.markdown("---")
                except Exception as e:
                    st.warning(f"Failed: {e}")

            if len(out_list) > 1:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, b in out_list: zf.writestr(fname, b)
                zip_buffer.seek(0)
                st.download_button(
                    f"Download ALL {noise_type} images (ZIP)",
                    data=zip_buffer,
                    file_name=f"{noise_type.lower().replace(' ', '')}_attacked_batch{param_short}_s{seed}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_zip_noise"
                )

# ----------------------------
# Main tabs
# ----------------------------
tabs = st.tabs(["Embed", "Extract", "Attack"])
with tabs[0]: tab_embed()
with tabs[1]: tab_extract()
with tabs[2]: tab_attack()
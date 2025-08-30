import io
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
import zipfile

import watermarking, utils, attacks, parameters

# Hide only the built-in "Send to Streamlit" download button in st_canvas
st.markdown("""
<style>
div[class*="st-drawable-canvas"] a[download] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# Small functions
# ============================================
def np_image_download_button(img_pil: Image.Image, label: str, filename: str, unique_key: str):
    buf = io.BytesIO()
    fmt = filename.split(".")[-1].upper()
    img_pil.save(buf, format="PNG" if fmt == "PNG" else fmt)
    st.download_button(
        label,
        data=buf.getvalue(),
        file_name=filename,
        mime="image/" + filename.split(".")[-1],
        use_container_width=False,
        key=unique_key  # Add the unique key here
    )

# ============================================
# Streamlit UI
# ============================================
st.set_page_config(page_title="Image Watermarking", layout="wide")
st.title("Image Watermarking")

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

    # Session flags
    if "embed_done" not in st.session_state:
        st.session_state["embed_done"] = False
    if "last_wm_bits" not in st.session_state:
        st.session_state["last_wm_bits"] = None
    if "last_wm_source" not in st.session_state:
        st.session_state["last_wm_source"] = None

    # ==== HOST IMAGES ====
    st.markdown("### Host Images")
    host_files = st.file_uploader(
        "Upload one or more images.",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="host_upl_multi",
    )

    host_pils: list[Image.Image] = []
    host_names: list[str] = []

    if host_files:
        for uf in host_files:
            try:
                pil = Image.open(uf).convert("RGB")
                host_pils.append(pil)
                host_names.append(getattr(uf, "name", "unnamed"))
            except Exception as e:
                st.warning(f"Failed to open {getattr(uf, 'name', 'file')}: {e}")

    # Previews for hosts
    if len(host_pils) == 0:
        st.info("No image(s) yet — upload one or more above to see previews.")
    else:
        st.markdown("**Image previews**")
        per_row = 3
        for i in range(0, len(host_pils), per_row):
            cols = st.columns(per_row, gap="medium")
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(host_pils):
                    break
                pil = host_pils[idx]
                name = host_names[idx]
                with col:
                    st.image(
                        pil,
                        use_container_width=True,
                        caption=f"{name} • {pil.width}×{pil.height}px",
                    )

    # ==== WATERMARK ====
    st.markdown("### Watermark Image")
    wm_source = st.radio("Choose watermark source", ["Upload watermark", "Draw watermark"], horizontal=True)

    wm_bits = None

    if wm_source == "Upload watermark":
        wm_file = st.file_uploader(
            "Upload watermark image (will be binarized to 32×32)",
            type=["png", "jpg", "jpeg", "webp"],
            key="wm_upl"
        )
        if wm_file is not None:
            wm_bits = utils.prepare_watermark_bits(Image.open(wm_file), size=parameters.WATERMARK_SIZE)

    else:  # Draw watermark
        st.info("Draw in black on a white canvas. The drawing is binarized to 32×32.")
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=8,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=256, width=256,
            drawing_mode="freedraw",
            key="wm_canvas",
        )
        if canvas and canvas.image_data is not None:
            wm_bits = utils.prepare_watermark_bits(Image.fromarray(canvas.image_data.astype(np.uint8)), size=parameters.WATERMARK_SIZE)

    # ==== WATERMARK PREVIEW (crisp 32×32, header only when present) ====
    bits_for_preview = None
    if wm_bits is not None:
        bits_for_preview = wm_bits
    elif st.session_state.get("last_wm_bits") is not None:
        bits_for_preview = st.session_state["last_wm_bits"]

    if bits_for_preview is not None:
        st.markdown("**Watermark preview (32×32, binary)**")
        canvas_size = 256  # match your canvas for a clear, pixelated upscale
        wm_bin_preview = np.array(bits_for_preview, dtype=np.uint8).reshape((parameters.WATERMARK_SIZE, parameters.WATERMARK_SIZE))
        big = Image.fromarray((wm_bin_preview * 255).astype(np.uint8), mode="L").resize(
            (canvas_size, canvas_size), Image.NEAREST
        )
        st.image(big, clamp=True)

        # Draw-only: show a download button for the exact 32×32 under the preview
        if wm_source == "Draw image":
            bin_img_exact = Image.fromarray((wm_bin_preview * 255).astype(np.uint8), mode="L")
            buf = BytesIO()
            bin_img_exact.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                "⬇️ Download watermark (32×32 PNG)",
                data=buf,
                file_name="watermark_32x32.png",
                mime="image/png",
                use_container_width=False,
            )
    else:
        st.info("No watermark yet — upload watermark to see preview.")

    # ==== EMBED ACTION ====
    # For Upload watermark → Embed button should appear below the preview.
    # For Draw watermark → also allow embedding here (keeps UX consistent).
    place_embed_button_here = True  # set True for both modes so it's always below preview

    if place_embed_button_here:
        if st.button("Embed Watermark"):
            if wm_bits is None or len(host_pils) == 0:
                st.warning("Please provide a watermark or at least one host image.")
                st.stop()

            # Convert all hosts to YCbCr arrays
            y_wm_list, y_list, cb_list, cr_list = [], [], [], []
            for hp in host_pils:
                y, cb, cr = utils.to_ycbcr_arrays(hp)
                y_list.append(y); cb_list.append(cb); cr_list.append(cr)
                y_wm_list.append(watermarking.embed_watermark(y, wm_bits))

            # Store last bits/source for preview after embed
            st.session_state["last_wm_bits"] = wm_bits
            st.session_state["last_wm_source"] = wm_source
            st.session_state["embed_done"] = True

            watermarked_files = []
            for idx, (hp, y_wm, cb, cr, name) in enumerate(zip(host_pils, y_wm_list, cb_list, cr_list, host_names), start=1):
                wm_rgb_pil = utils.from_ycbcr_arrays(y_wm, cb, cr)
                out_name = f"{name.rsplit('.',1)[0]}_watermarked.png"
                watermarked_files.append((out_name, wm_rgb_pil))

                wm_rgb = utils.pil_to_np_rgb(wm_rgb_pil)
                host_rgb = utils.pil_to_np_rgb(hp)

                unique_key = f"download_{name}_{idx}"  # Unique key for each download button

                # Save last item for Attack tab
                st.session_state["watermarked_rgb"] = wm_rgb

                # Capacity
                H, W = hp.size[1], hp.size[0]

                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.markdown("**Original**")
                    st.image(hp, use_container_width=True)
                with col_b:
                    st.markdown("**Watermarked**")
                    st.image(wm_rgb_pil, use_container_width=True)

                psnr_val = utils.psnr(host_rgb, wm_rgb)
                st.metric("PSNR (Original vs Watermarked)", f"{psnr_val:.2f} dB")
                np_image_download_button(
                    wm_rgb_pil,
                    f"⬇️ Download `{name}` (watermarked)",
                    out_name,
                    unique_key=unique_key  # Provide the unique key here
                )

                st.markdown("---")

            # ---- Download ALL as a ZIP ----
            if len(watermarked_files) > 1:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, pil_img in watermarked_files:
                        img_bytes = BytesIO()
                        pil_img.save(img_bytes, format="PNG")
                        img_bytes.seek(0)
                        zf.writestr(fname, img_bytes.read())
                zip_buffer.seek(0)
                st.download_button(
                    "⬇️ Download ALL watermarked images (ZIP)",
                    data=zip_buffer,
                    file_name="watermarked_batch.zip",
                    mime="application/zip",
                    use_container_width=True,
                )


# ==========================================================
# TAB 2: EXTRACT
# ==========================================================
with tabs[1]:
    st.header("Extract")

    # ==== Watermarked Image upload ====
    st.markdown("### Watermarked Image")
    attacked_files = st.file_uploader(
        "Upload one or more watermarked images.",
        type=["png","jpg","jpeg","webp"],
        key="attacked_upl",
        accept_multiple_files=True
    )
    if not attacked_files:
        st.info("No watermarked images yet — upload one or more above to see previews.")

    # ==== (PREVIEW FIRST) Watermarked images preview (grid) ====
    atk_pils_preview, atk_names_preview = [], []
    if attacked_files:
        st.markdown("**Watermarked image previews**")
        for f in attacked_files:
            try:
                p = Image.open(f).convert("RGB")
                atk_pils_preview.append(p)
                atk_names_preview.append(getattr(f, "name", "watermarked.png"))
            except Exception as e:
                st.warning(f"Failed to open {getattr(f,'name','file')}: {e}")

        # Grid layout
        per_row = 3
        for i in range(0, len(atk_pils_preview), per_row):
            cols = st.columns(per_row, gap="medium")
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(atk_pils_preview):
                    break
                with col:
                    pil = atk_pils_preview[idx]
                    name = atk_names_preview[idx]
                    st.image(
                        pil,
                        use_container_width=True,
                        caption=f"{name} • {pil.width}×{pil.height}px",
                    )

    # ==== Original watermark upload (SECOND) ====
    st.markdown("### Watermark (for compute Bit Error Rate)")
    ori_wm_file = st.file_uploader(
        "Upload original watermark image .",
        type=["png","jpg","jpeg","webp"],
        key="oriwm_upl"
    )
    if not ori_wm_file:
        st.info("No watermark yet — upload watermark to see preview.")

    # Original watermark preview (crisp 32×32 if provided)
    ref_bits = None
    if ori_wm_file:
        try:
            st.markdown("**Original watermark (32×32 binarized preview)**")
            ori_pil = Image.open(ori_wm_file).convert("RGBA")
            # composite on white to handle transparency
            bg = Image.new("RGBA", ori_pil.size, (255, 255, 255, 255))
            ori_rgb = Image.alpha_composite(bg, ori_pil).convert("RGB")

            # binarize to 32×32 using your helper
            ori_wm_bin = utils.prepare_watermark_bits(ori_rgb, size=parameters.WATERMARK_SIZE)
            ref_bits = ori_wm_bin.reshape(-1).tolist()

            # crisp preview (nearest neighbor upscale)
            canvas_size = 256
            big = Image.fromarray((ori_wm_bin * 255).astype(np.uint8), mode="L").resize(
                (canvas_size, canvas_size), Image.NEAREST
            )
            st.image(big, clamp=True)
        except Exception as e:
            st.warning(f"Failed to read original watermark: {e}")
            ref_bits = None

    # ---------- EXTRACT ACTION ----------
    if st.button("Extract Watermark(s)"):
        if not attacked_files:
            st.warning("Please upload at least one watermarked image.")
            st.stop()

        # Build lists of Y from all attacked files
        bits_list, atk_pils, atk_names, y_list = [], [], [], []
        for f in attacked_files:
            pil_img = Image.open(f).convert("RGB")
            atk_pils.append(pil_img)
            atk_names.append(getattr(f, "name", "watermarked.png"))
            y, _, _ = utils.to_ycbcr_arrays(pil_img)
            y_list.append(y)
            # extract bits
            bits_list.append(watermarking.extract_watermark(y))

        # Show per-file results
        for idx, (name, bits_out) in enumerate(zip(atk_names, bits_list), start=1):
            wm_rec = np.array(bits_out, dtype=np.uint8).reshape(parameters.WATERMARK_SIZE, parameters.WATERMARK_SIZE)
            st.markdown(f"### Extracted #{idx}: `{name}`")
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown("**Source image**")
                st.image(atk_pils[idx-1], use_container_width=True)
            with col2:
                st.markdown("**Recovered watermark**")
                # crisp recovered preview
                rec_big = Image.fromarray((wm_rec * 255).astype(np.uint8), mode="L").resize(
                    (256, 256), Image.NEAREST
                )
                st.image(rec_big, clamp=True)

            if ref_bits is not None:
                ber_val, stats = utils.bit_error_rate(ref_bits, bits_out, return_counts=True)
                st.metric("BER", f"{ber_val:.4f}")
                st.write(f"Accuracy: **{stats['accuracy']:.4f}**  |  Errors: **{stats['errors']}/{stats['total']}**")
            else:
                st.info("Upload the original watermark image above to compute Bit Error Rate.")
            st.markdown("---")


# ==========================================================
# TAB 3: ATTACK
# ==========================================================
with tabs[2]:
    st.header("Attack")

    # ---------- Attack Method Tabs ----------
    attack_tabs = st.tabs(["JPEG", "Cropping"])

    # JPEG Attack Tab
    with attack_tabs[0]:
        st.markdown("### JPEG Attack")
        
        # Upload attacked images
        jpeg_files = st.file_uploader(
            "Upload one or more original watermarked images",
            type=["png", "jpg", "jpeg", "webp"],
            key="jpeg_attack_files",
            accept_multiple_files=True,
        )
        
        if not jpeg_files:
            st.info("Upload original watermarked images to see JPEG attack preview.")
        else:
            # JPEG Quality Slider
            jpeg_quality = st.slider("JPEG Quality", 10, 95, 75, step=1, key="jpeg_quality_slider")
            
            st.markdown("**JPEG Attack Preview**")
            
            # Process uploaded files
            jpeg_pils, jpeg_names = [], []
            for f in jpeg_files:
                try:
                    p = Image.open(f).convert("RGB")
                    jpeg_pils.append(p)
                    jpeg_names.append(getattr(f, "name", "original.png"))
                except Exception as e:
                    st.warning(f"Failed to open {getattr(f,'name','file')}: {e}")
            
            # Store for batch download
            jpeg_download_files = []
            
            # Show before/after previews
            for idx, (pil, name) in enumerate(zip(jpeg_pils, jpeg_names)):
                # Apply JPEG compression with current quality
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=False)
                buf.seek(0)
                jpeg_attacked_pil = Image.open(buf).convert("RGB")
                
                st.markdown(f"#### Image {idx + 1}: `{name}`")
                
                # Show original vs JPEG attacked side by side
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.markdown("**Original**")
                    st.image(
                        pil,
                        use_container_width=True,
                        caption=f"Original • {pil.width}×{pil.height}px"
                    )
                with col2:
                    st.markdown(f"**JPEG Quality {jpeg_quality}**")
                    st.image(
                        jpeg_attacked_pil,
                        use_container_width=True,
                        caption=f"JPEG Q{jpeg_quality} • {jpeg_attacked_pil.width}×{jpeg_attacked_pil.height}px"
                    )
                
                # Individual download button for this quality
                out_name = f"{name.rsplit('.', 1)[0]}_jpeg_q{jpeg_quality}.jpg"
                np_image_download_button(
                    jpeg_attacked_pil,
                    f"⬇️ Download JPEG Q{jpeg_quality}",
                    out_name,
                    unique_key=f"jpeg_download_{idx}_{jpeg_quality}"
                )
                
                jpeg_download_files.append((out_name, jpeg_attacked_pil))
                
                if idx < len(jpeg_pils) - 1:  # Add separator except for last image
                    st.markdown("---")
            
            # Batch download if multiple files
            if len(jpeg_download_files) > 1:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, pil_img in jpeg_download_files:
                        img_bytes = BytesIO()
                        pil_img.save(img_bytes, format="JPEG")
                        img_bytes.seek(0)
                        zf.writestr(fname, img_bytes.read())
                zip_buffer.seek(0)
                st.download_button(
                    f"⬇️ Download ALL JPEG Q{jpeg_quality} images (ZIP)",
                    data=zip_buffer,
                    file_name=f"jpeg_attacked_batch_q{jpeg_quality}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    # Cropping Attack Tab
    with attack_tabs[1]:
        st.markdown("### Cropping Attack")
        
        # Upload attacked images
        cropping_files = st.file_uploader(
            "Upload one or more original watermarked images",
            type=["png", "jpg", "jpeg", "webp"],
            key="cropping_attack_files",
            accept_multiple_files=True,
        )
        
        if not cropping_files:
            st.info("Upload original watermarked images to see cropping attack preview.")
        else:
            # Cropping parameter sliders
            area_ratio = st.slider("Area Ratio", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f", key="crop_area_ratio")
            num_patches = st.slider("Number of Patches", 1, 10, 2, key="crop_num_patches")
            seed = st.slider("Seed", 0, 100, 42, key="crop_seed")
            
            st.markdown("**Cropping Attack Preview**")
            
            # Process uploaded files
            cropping_pils, cropping_names = [], []
            for f in cropping_files:
                try:
                    p = Image.open(f).convert("RGB")
                    cropping_pils.append(p)
                    cropping_names.append(getattr(f, "name", "original.png"))
                except Exception as e:
                    st.warning(f"Failed to open {getattr(f,'name','file')}: {e}")
            
            # Store for batch download
            cropping_download_files = []
            
            # Show before/after previews
            for idx, (pil, name) in enumerate(zip(cropping_pils, cropping_names)):
                # Apply cropping attack with current parameters
                src_rgb = np.array(pil)
                cropping_attacked_pil = attacks.crop_attack(
                    in_arr=src_rgb.copy(), 
                    area_ratio=area_ratio, 
                    num_patches=num_patches, 
                    seed=seed
                )
                
                st.markdown(f"#### Image {idx + 1}: `{name}`")
                
                # Show original vs cropping attacked side by side
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.markdown("**Original**")
                    st.image(
                        pil,
                        use_container_width=True,
                        caption=f"Original • {pil.width}×{pil.height}px"
                    )
                with col2:
                    st.markdown(f"**Cropped (Area: {area_ratio:.4f}, Patches: {num_patches}, Seed: {seed})**")
                    st.image(
                        cropping_attacked_pil,
                        use_container_width=True,
                        caption=f"Cropped • {cropping_attacked_pil.width}×{cropping_attacked_pil.height}px"
                    )
                
                # Individual download button for this configuration
                out_name = f"{name.rsplit('.', 1)[0]}_cropped_a{area_ratio}_p{num_patches}_s{seed}.png"
                np_image_download_button(
                    cropping_attacked_pil,
                    f"⬇️ Download Cropped",
                    out_name,
                    unique_key=f"cropping_download_{idx}_{area_ratio}_{num_patches}_{seed}"
                )
                
                cropping_download_files.append((out_name, cropping_attacked_pil))
                
                if idx < len(cropping_pils) - 1:  # Add separator except for last image
                    st.markdown("---")
            
            # Batch download if multiple files
            if len(cropping_download_files) > 1:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fname, pil_img in cropping_download_files:
                        img_bytes = BytesIO()
                        pil_img.save(img_bytes, format="PNG")
                        img_bytes.seek(0)
                        zf.writestr(fname, img_bytes.read())
                zip_buffer.seek(0)
                st.download_button(
                    f"⬇️ Download ALL Cropped images (ZIP)",
                    data=zip_buffer,
                    file_name=f"cropped_attacked_batch_a{area_ratio}_p{num_patches}_s{seed}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

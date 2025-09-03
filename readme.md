# Tag It Smart: Image Watermarking System

Tag It Smart is a robust digital image watermarking system that embeds and extracts binary watermarks using DWT-DCT-SVD transform domain techniques with Quantization Index Modulation (QIM) to withstand compression and cropping from WhatsApp.

## 🎯 Project Overview

This project implements a comprehensive watermarking solution that can:
- **Embed** binary watermarks into host images with minimal visual distortion
- **Extract** watermarks from watermarked images, even after various attacks
- **Simulate attacks** like JPEG compression, cropping, rotation, and noise
- **Analyze robustness** through automated testing and metrics

## ✨ Key Features

### Watermarking Algorithms
- **Standard Algorithm**: Basic DWT-DCT-SVD watermarking with QIM
- **Enhanced Algorithm**: Rotation-resistant extraction with angle search optimization

### Attack Simulation
- JPEG compression (configurable quality)
- Spatial cropping attacks
- Small rotation attacks
- Gaussian and uniform noise attacks

### Performance Metrics
- **PSNR**: Peak Signal-to-Noise Ratio for image quality assessment
- **BER**: Bit Error Rate for watermark accuracy
- **NCC**: Normalized Cross Correlation for watermark similarity

### Interactive Web Interface
- Streamlit-based GUI for easy watermark embedding and extraction
- Canvas drawing tool for custom watermark creation
- Batch processing and ZIP download support

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Required packages (install via pip):

```bash
pip install numpy pandas matplotlib pillow opencv-python
pip install pywavelets scipy streamlit streamlit-drawable-canvas
```

### Quick Start
1. Clone or download the project files
2. Install dependencies: `pip install -r requirements.txt` (if available)
3. Run the web app: `streamlit run app.py`

## 📁 Project Structure

```
d:\ip-app\
├── app.py                 # Streamlit web interface
├── watermarking.py        # Core watermarking algorithms
├── utilities.py           # Helper functions and metrics
├── attacks.py             # Attack simulation functions
├── parameters.py          # System configuration
├── experiment.ipynb       # Experimental analysis notebook
├── readme.md             # This file
└── dataset/              # Data directories (created automatically)
    ├── host/             # Original host images
    ├── watermarked/      # Watermarked images
    ├── jpeg_q75/         # JPEG attacked images
    ├── cropped/          # Cropped images
    └── analysis/         # Experimental results
```

## 🚀 Usage

### Web Interface (Recommended)
```bash
streamlit run app.py
```

The web interface provides three main tabs:
1. **Embed**: Upload host images and create/upload watermarks
2. **Extract**: Extract watermarks from images and compute metrics
3. **Attack**: Simulate various attacks on watermarked images

### Programmatic Usage

#### Basic Embedding
```python
import watermarking, utilities
from PIL import Image

# Load host image and watermark
host = Image.open("host.jpg").convert("RGB")
watermark = Image.open("watermark.png")

# Convert to YCbCr and prepare watermark
Y, Cb, Cr = utilities.to_ycbcr_arrays(host)
wm_bits = utilities.prepare_watermark_bits(watermark, 32)

# Embed watermark
Y_wm = watermarking.embed_watermark(Y, wm_bits)

# Convert back to RGB
result = utilities.from_ycbcr_arrays(Y_wm, Cb, Cr)
result.save("watermarked.png")
```

#### Basic Extraction
```python
# Extract using standard algorithm
extracted_bits = watermarking.extract_watermark(Y_wm)

# Extract using enhanced algorithm (rotation-resistant)
extracted_bits_enh = watermarking.extract_watermark_enhanced(Y_wm)

# Calculate metrics
ber = utilities.bit_error_rate(original_bits, extracted_bits)
ncc = utilities.normalized_cross_correlation(original_bits, extracted_bits)
```

## ⚙️ Configuration

Key parameters in `parameters.py`:

```python
DWT_LEVELS = 2           # Wavelet decomposition levels
DCT_BLOCK_SIZE = 8       # DCT block size
QIM_STEP = 100.0         # Quantization step for QIM
WATERMARK_SIZE = 32      # Watermark dimensions (32×32)
WAVELET_NAME = 'haar'    # Wavelet type
```

## 🔬 Experimental Analysis

The `experiment.ipynb` notebook provides comprehensive testing:

### Test Scenarios
- **Clean extraction**: Baseline performance
- **WhatsApp compression**: Real-world social media scenario
- **JPEG compression**: Quality factor 75
- **Cropping attacks**: 5% area removal
- **Rotation attacks**: 1-degree rotation
- **Noise attacks**: Gaussian (σ=10) and uniform (±20)

### Metrics Analysis
- Individual image performance across all attacks
- Comparative analysis between standard and enhanced algorithms
- Statistical summaries and visualizations

### Algorithm Strengths
- **Standard**: Fast, reliable for non-geometric attacks
- **Enhanced**: Superior performance under rotation attacks

## 🔧 Customization

### Adding New Attacks
1. Implement attack function in `attacks.py`
2. Add UI controls in `app.py`
3. Include in experimental analysis

### Modifying Watermark Size
1. Update `WATERMARK_SIZE` in `parameters.py`
2. Ensure consistent usage across all modules

### Changing Transform Parameters
- Adjust `DWT_LEVELS`, `QIM_STEP` for different robustness/capacity trade-offs
- Modify `WAVELET_NAME` for different wavelets (e.g., 'db4', 'bior2.2')

## 📝 Notes

### Image Requirements
- **Size**: 512×1080 pixels (automatically resized)
- **Format**: PNG, JPEG, WebP supported
- **Aspect ratio**: Max 3:1 (extremely elongated images rejected)

### Watermark Requirements
- Automatically binarized using Otsu thresholding
- Resized to 32×32 pixels
- Black pixels become 1-bits, white pixels become 0-bits

## 🤝 Contributing

To extend this project:
1. Follow the existing code structure
2. Add comprehensive error handling
3. Include unit tests for new functions
4. Update documentation accordingly

## 📄 License

This project is provided for educational and research purposes. Please cite appropriately if used in academic work.


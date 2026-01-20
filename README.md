# Flesh and Blood Card Scanner

A Python-based card scanning and identification system for Flesh and Blood TCG cards.

## Features

- **Live Webcam Capture**: Real-time card capture with alignment feedback and auto-capture
- **Card Detection**: Automatic card boundary detection using OpenCV contour detection
- **Image Enhancement**: Advanced preprocessing with CLAHE contrast enhancement and sharpening
- **OCR Text Extraction**: Extracts text from cards using Tesseract OCR (supports English and Japanese)
- **Card Vault API Integration**: Searches the official Flesh and Blood Card Vault API for card identification
- **Batch Processing**: Process multiple card images at once
- **Detailed Output**: Comprehensive card information including stats, artist, rules text, and image URLs

## Setup

### Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-jpn

# macOS
brew install tesseract tesseract-lang
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Webcam Capture (Recommended)

**Best for:** Capturing high-quality images for accurate identification

1. **Set up your capture station** (see [WEBCAM_SETUP.md](WEBCAM_SETUP.md))
   - Position webcam 12-18 inches above a white surface
   - Ensure even lighting from both sides
   - Camera should point straight down (90° angle)

2. **Start the webcam capture**:
```bash
python webcam_capture.py
```

3. **Capture cards**:
   - Place card in view, align with target rectangle
   - Auto-capture triggers when card is properly positioned
   - Or press SPACEBAR for manual capture
   - Press Q to quit

4. **Process captured cards**:
```bash
python process_captured_cards.py
```

This will identify all cards captured during your session.

### Method 2: Scan Existing Images

**Best for:** Processing photos you already have

#### Single Card:
```bash
python scan_card.py <image_path> [output_dir]
```

Example:
```bash
python scan_card.py "Screenshot 2025-05-11 22-15-18.png" output
```

#### Batch Process:
```bash
python test_all_cards.py
```

This will process all PNG images in the current directory.

## How It Works

1. **Detection**: Identifies card boundaries using contour detection
2. **Extraction**: Crops and extracts the card from the image
3. **Enhancement**: Applies contrast enhancement and sharpening for better OCR
4. **OCR**: Extracts text from the card image
5. **API Search**: Searches the Card Vault API using extracted text
6. **Matching**: Finds the best matching card using similarity scoring

## API Reference

The scanner uses the official Flesh and Blood Card Vault API:
- **Base URL**: `https://api.cardvault.fabtcg.com/carddb/api/v1/advanced-search/`
- **Search**: Supports keyword-based card searching
- **Results**: Returns comprehensive card data including stats, images, and metadata

## Tips for Best Results

- ✓ **Use the webcam capture system** - provides consistent, high-quality images
- ✓ **Proper lighting** - even lighting from multiple angles eliminates shadows
- ✓ **90° camera angle** - camera directly overhead (not at an angle)
- ✓ **Remove from sleeves** - reduces glare and improves OCR accuracy
- ✓ **White background** - increases contrast for better card detection
- ✓ **Hold still** - let auto-capture trigger when card is perfectly aligned

## Limitations

- OCR accuracy depends on image quality, lighting, and camera angle
- Japanese cards may have lower identification rates when matched against English API
- Print ID extraction works best with clear, overhead photos
- Cards at angles or with poor lighting may not be identified correctly

## Test Images

The repository includes 5 test images of Flesh and Blood cards (Japanese versions) for testing the scanner.

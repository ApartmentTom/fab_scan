# Flesh and Blood Card Scanner

A Python-based card scanning and identification system for Flesh and Blood TCG cards.

## Features

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

### Scan a Single Card

```bash
python scan_card.py <image_path> [output_dir]
```

Example:
```bash
python scan_card.py "Screenshot 2025-05-11 22-15-18.png" output
```

### Batch Process All Images

```bash
python test_all_cards.py
```

This will process all PNG images in the current directory and save enhanced images to the `output/` folder.

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

## Limitations

- OCR accuracy depends on image quality and lighting
- Japanese cards may have lower identification rates when matched against English API
- Best results with well-lit, clear images

## Test Images

The repository includes 5 test images of Flesh and Blood cards (Japanese versions) for testing the scanner.

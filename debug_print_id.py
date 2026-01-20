#!/usr/bin/env python3
"""
Debug script to visualize print ID extraction
"""

import cv2
import sys
from scan_card import CardScanner

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_print_id.py <image_path>")
        return

    image_path = sys.argv[1]
    scanner = CardScanner()

    # Load and process image
    image = scanner.load_image(image_path)
    detection = scanner.detect_card(image)
    card_image = scanner.extract_card_image(image, detection)
    enhanced = scanner.enhance_card_image(card_image)

    # Extract print ID region
    bottom_region = scanner.extract_print_id_region(enhanced)
    processed = scanner.preprocess_for_print_id(bottom_region)

    # Save debug images
    cv2.imwrite('debug_bottom_region.png', bottom_region)
    cv2.imwrite('debug_processed.png', processed)

    print(f"Saved debug images:")
    print(f"  debug_bottom_region.png - Raw bottom region")
    print(f"  debug_processed.png - Preprocessed for OCR")

    # Try OCR
    import pytesseract
    from PIL import Image as PILImage

    pil_image = PILImage.fromarray(processed)

    # Try different OCR configs
    print("\nTrying different OCR configurations:")

    configs = [
        ('Default', ''),
        ('PSM 7', '--psm 7'),
        ('PSM 6', '--psm 6'),
        ('PSM 11', '--psm 11'),
        ('With whitelist', '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'),
    ]

    for name, config in configs:
        try:
            text = pytesseract.image_to_string(pil_image, lang='eng', config=config)
            print(f"\n{name}: '{text.strip()}'")
        except Exception as e:
            print(f"\n{name}: Error - {e}")

if __name__ == "__main__":
    main()

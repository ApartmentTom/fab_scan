#!/usr/bin/env python3
"""
Test full card OCR on the English card
"""

import cv2
import sys
from scan_card import CardScanner
from PIL import Image
import pytesseract

def main():
    image_path = "Screenshot 2025-05-11 22-15-46.png"
    scanner = CardScanner()

    # Load and process image
    image = scanner.load_image(image_path)
    detection = scanner.detect_card(image)
    card_image = scanner.extract_card_image(image, detection)
    enhanced = scanner.enhance_card_image(card_image)

    # Convert to PIL
    rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    print("Extracting text from full card with multiple PSM modes:\n")

    configs = [
        ('PSM 3 - Auto', '--psm 3'),
        ('PSM 6 - Block', '--psm 6'),
        ('PSM 4 - Single column', '--psm 4'),
    ]

    for name, config in configs:
        print(f"\n{name}:")
        print("="*60)
        try:
            text = pytesseract.image_to_string(pil_image, lang='eng', config=config)
            print(text[:500] if len(text) > 500 else text)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

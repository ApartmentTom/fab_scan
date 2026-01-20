#!/usr/bin/env python3
"""
Extract just the card name from top region
"""

import cv2
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

    # Detect actual card boundaries
    card_rect = scanner.detect_card_boundaries(enhanced)

    if card_rect:
        x, y, w, h = card_rect
        # Extract top 15% where card name should be
        name_region = enhanced[y:y+int(h*0.15), x:x+w]

        # Save for inspection
        cv2.imwrite('debug_name_region.png', name_region)
        print("Saved debug_name_region.png")

        # Convert to PIL and run OCR
        rgb = cv2.cvtColor(name_region, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        print("\nTrying to extract card name:")
        configs = [
            '--psm 7',  # Single line
            '--psm 8',  # Single word
            '--psm 6',  # Block
        ]

        for config in configs:
            try:
                text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
                cleaned = ' '.join(text.split())
                if cleaned:
                    print(f"{config}: '{cleaned}'")
            except:
                pass
    else:
        print("Could not detect card boundaries")

if __name__ == "__main__":
    main()

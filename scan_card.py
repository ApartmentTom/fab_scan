#!/usr/bin/env python3
"""
Flesh and Blood Card Scanner and Identifier
"""

import cv2
import numpy as np
from PIL import Image
import sys
import os


class CardScanner:
    """Main card scanning and identification class"""

    def __init__(self):
        self.min_card_area = 50000  # Minimum area for card detection

    def load_image(self, image_path):
        """Load an image from file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    def preprocess_image(self, image):
        """Preprocess image for card detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def detect_card(self, image):
        """Detect card boundaries in the image"""
        # Preprocess
        thresh = self.preprocess_image(image)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find the largest rectangular contour (likely the card)
        largest_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > self.min_card_area:
                # Approximate the contour to a polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Check if it's roughly rectangular (4 corners)
                if len(approx) >= 4:
                    max_area = area
                    largest_contour = contour

        if largest_contour is None:
            return None

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        return {
            'contour': largest_contour,
            'bbox': (x, y, w, h),
            'area': max_area
        }

    def extract_card_image(self, image, detection):
        """Extract and straighten the card from the image"""
        if detection is None:
            return image

        x, y, w, h = detection['bbox']

        # Crop the card with some padding
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        card_image = image[y1:y2, x1:x2]

        return card_image

    def enhance_card_image(self, card_image):
        """Enhance card image for better identification"""
        # Resize to standard size for consistency
        standard_height = 800
        aspect_ratio = card_image.shape[1] / card_image.shape[0]
        standard_width = int(standard_height * aspect_ratio)

        resized = cv2.resize(
            card_image, (standard_width, standard_height),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Enhance contrast
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def identify_card(self, card_image):
        """Identify the card from the image"""
        # For now, return basic image statistics
        # This is where you would implement:
        # - OCR for text extraction
        # - Feature matching against a card database
        # - API calls to card databases

        height, width = card_image.shape[:2]
        avg_color = cv2.mean(card_image)[:3]

        return {
            'dimensions': (width, height),
            'avg_color': avg_color,
            'status': 'detected'
        }

    def scan(self, image_path, output_dir=None):
        """Main scanning pipeline"""
        print(f"Loading image: {image_path}")
        image = self.load_image(image_path)

        print("Detecting card...")
        detection = self.detect_card(image)

        if detection:
            print(f"Card detected! Area: {detection['area']:.0f} pixels")
            print(f"Bounding box: {detection['bbox']}")
        else:
            print("No card detected, using full image")

        print("Extracting card image...")
        card_image = self.extract_card_image(image, detection)

        print("Enhancing image...")
        enhanced = self.enhance_card_image(card_image)

        print("Identifying card...")
        identification = self.identify_card(enhanced)

        # Save processed image if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            basename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"processed_{basename}")
            cv2.imwrite(output_path, enhanced)
            print(f"Saved processed image: {output_path}")

        return {
            'detection': detection,
            'identification': identification,
            'processed_image': enhanced
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_card.py <image_path> [output_dir]")
        print("\nExample:")
        print("  python scan_card.py 'Screenshot 2025-05-11 22-15-18.png'")
        print("  python scan_card.py 'Screenshot 2025-05-11 22-15-18.png' output/")
        return

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    scanner = CardScanner()

    try:
        result = scanner.scan(image_path, output_dir)
        print("\n" + "="*50)
        print("SCAN COMPLETE")
        print("="*50)
        print(f"Identification: {result['identification']}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

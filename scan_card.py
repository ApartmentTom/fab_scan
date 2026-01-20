#!/usr/bin/env python3
"""
Flesh and Blood Card Scanner and Identifier
"""

import cv2
import numpy as np
from PIL import Image
import sys
import os
import requests
import re
from difflib import SequenceMatcher


class CardScanner:
    """Main card scanning and identification class"""

    def __init__(self):
        self.min_card_area = 50000  # Minimum area for card detection
        self.api_base_url = "https://api.cardvault.fabtcg.com/carddb/api/v1/advanced-search/"

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

    def detect_card_boundaries(self, card_image):
        """Detect the actual card boundaries within the image using edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # If no contours found, return None (will use full image)
            return None

        # Find the largest rectangular contour
        largest_rect = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Check if it's roughly card-shaped (aspect ratio between 1.3 and 1.6)
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 1.8 and area > max_area:
                    max_area = area
                    largest_rect = (x, y, w, h)

        return largest_rect

    def extract_print_id_region(self, card_image):
        """Extract the bottom region where print ID typically appears"""
        # Try to detect actual card boundaries
        card_rect = self.detect_card_boundaries(card_image)

        if card_rect:
            x, y, w, h = card_rect
            # Extract bottom 15% of the detected card
            print_id_start = y + int(h * 0.82)
            print_id_end = y + int(h * 0.95)
            print_id_left = x + int(w * 0.05)
            print_id_right = x + int(w * 0.95)

            bottom_region = card_image[print_id_start:print_id_end, print_id_left:print_id_right]
        else:
            # Fallback to percentage-based extraction
            height, width = card_image.shape[:2]
            bottom_start = int(height * 0.45)
            bottom_end = int(height * 0.55)
            width_end = int(width * 0.5)
            bottom_region = card_image[bottom_start:bottom_end, :width_end]

        return bottom_region

    def preprocess_for_print_id(self, region):
        """Specialized preprocessing for print ID extraction"""
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply thresholding to get black text on white background
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (we want black text on white background)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # Resize to improve OCR accuracy
        scale_factor = 3
        height, width = binary.shape
        resized = cv2.resize(binary, (width * scale_factor, height * scale_factor),
                            interpolation=cv2.INTER_CUBIC)

        return resized

    def extract_print_id(self, card_image):
        """Extract print ID from card image"""
        try:
            import pytesseract
        except ImportError:
            return None

        # Extract bottom region
        bottom_region = self.extract_print_id_region(card_image)

        # Save debug image (optional)
        # cv2.imwrite('debug_print_region.png', bottom_region)

        # Preprocess for better OCR
        processed = self.preprocess_for_print_id(bottom_region)

        # Convert to PIL Image
        pil_image = Image.fromarray(processed)

        # Try multiple OCR configurations for best results
        ocr_results = []

        configs = [
            '--psm 6',  # Assume uniform block of text
            '--psm 7',  # Treat as single text line
            '--psm 11',  # Sparse text
        ]

        for config in configs:
            try:
                # Try with alphanumeric whitelist
                config_with_whitelist = f'{config} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                text = pytesseract.image_to_string(pil_image, lang='eng', config=config_with_whitelist)
                if text.strip():
                    ocr_results.append(text)

                # Also try without whitelist to catch variations
                text_no_whitelist = pytesseract.image_to_string(pil_image, lang='eng', config=config)
                if text_no_whitelist.strip():
                    ocr_results.append(text_no_whitelist)

            except Exception as e:
                continue

        # Look for print ID patterns in all OCR results
        # Pattern: Optional prefix, 2-4 letters, 3-4 digits
        patterns = [
            r'\b([A-Z]{2,4}\d{3,4})\b',  # ELE201, MST001, LORE143
            r'\b([A-Z]-[A-Z]{2,4}\d{3,4})\b',  # U-ELE074
            r'\b([A-Z]{2}_[A-Z]{2,4}\d{3,4})\b',  # DE_ELE201, FR_MST001
            r'\b([A-Z]{3}\d{3})\b',  # Simplified: 3 letters + 3 digits
        ]

        for text in ocr_results:
            # Clean up text
            cleaned_text = text.upper().replace('O', '0').replace('I', '1')  # Common OCR mistakes

            for pattern in patterns:
                matches = re.findall(pattern, cleaned_text)
                if matches:
                    print_id = matches[0]
                    print(f"Found print ID: {print_id}")
                    return print_id

        # If no pattern match but we have some alphanumeric text, return it for debugging
        for text in ocr_results:
            cleaned = re.sub(r'[^A-Z0-9_-]', '', text.upper().strip())
            if 5 <= len(cleaned) <= 15:  # Reasonable length for a print ID
                print(f"Possible print ID (no pattern match): {cleaned}")
                return cleaned

        return None

    def extract_text_ocr(self, card_image):
        """Extract text from card image using OCR"""
        try:
            import pytesseract
        except ImportError:
            print("Warning: pytesseract not available, skipping OCR")
            return ""

        # Convert to PIL Image
        rgb_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Extract text - try with different languages
        try:
            # Try English first
            text_en = pytesseract.image_to_string(pil_image, lang='eng')

            # Try Japanese if available
            try:
                text_ja = pytesseract.image_to_string(pil_image, lang='jpn')
                text = text_en + "\n" + text_ja
            except:
                text = text_en

        except Exception as e:
            print(f"OCR error: {e}")
            return ""

        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def search_card_api(self, query_text, print_id=None):
        """Search for card using the Card Vault API"""
        results = []

        # If we have a print ID, search for it first (most reliable)
        if print_id:
            print(f"Searching by print ID: {print_id}")
            try:
                response = requests.get(
                    self.api_base_url,
                    params={'q': print_id},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        results.extend(data['results'])
                        # If we got results from print ID, return immediately
                        if len(results) > 0:
                            print(f"Found {len(results)} matches by print ID")
                            return results

            except Exception as e:
                print(f"API search error for print ID: {e}")

        # Fall back to text-based search
        if not query_text:
            return results

        # Extract potential card name keywords from text
        # Remove common words and keep significant terms
        words = query_text.split()
        keywords = [w for w in words if len(w) >= 3][:5]  # Take first 5 significant words

        # Try searching with different keyword combinations
        search_terms = []
        if keywords:
            search_terms.append(' '.join(keywords[:2]))  # First 2 words
            if len(keywords) >= 1:
                search_terms.append(keywords[0])  # Just first word

        for search_term in search_terms:
            try:
                response = requests.get(
                    self.api_base_url,
                    params={'q': search_term},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        results.extend(data['results'][:10])  # Get top 10 results
                        if len(results) >= 5:  # Stop if we have enough results
                            break

            except Exception as e:
                print(f"API search error: {e}")
                continue

        return results

    def find_best_match(self, ocr_text, api_results):
        """Find the best matching card from API results"""
        if not api_results:
            return None

        best_match = None
        best_score = 0

        ocr_lower = ocr_text.lower()

        for card in api_results:
            card_name = card.get('printed_name', '')
            card_text = card.get('printed_rules_text', '')
            card_type = card.get('printed_typebox', '')

            # Calculate similarity score
            name_similarity = SequenceMatcher(None, ocr_lower, card_name.lower()).ratio()

            # Check if card name appears in OCR text
            name_in_text = 1.0 if card_name.lower() in ocr_lower else 0.0

            # Combined score
            score = (name_similarity * 0.5) + (name_in_text * 0.5)

            if score > best_score:
                best_score = score
                best_match = card

        return {
            'card': best_match,
            'confidence': best_score
        }

    def identify_card(self, card_image):
        """Identify the card from the image using OCR and API"""
        height, width = card_image.shape[:2]

        # Try to extract print ID first (most reliable)
        print("Extracting print ID...")
        print_id = self.extract_print_id(card_image)

        if print_id:
            print(f"Detected print ID: {print_id}")
        else:
            print("No print ID detected")

        # Extract general text for fallback
        print("Extracting text with OCR...")
        ocr_text = self.extract_text_ocr(card_image)

        if ocr_text:
            print(f"Extracted text: {ocr_text[:100]}...")  # Show first 100 chars
        else:
            print("No text extracted from image")

        # Search API with print ID (if available) and text
        print("Searching Card Vault API...")
        api_results = self.search_card_api(ocr_text, print_id=print_id)

        if api_results:
            print(f"Found {len(api_results)} potential matches")
        else:
            print("No matches found in API")

        # Find best match
        match = self.find_best_match(ocr_text, api_results)

        result = {
            'dimensions': (width, height),
            'ocr_text': ocr_text,
            'detected_print_id': print_id,
            'api_results_count': len(api_results),
        }

        if match and match['card']:
            card = match['card']
            result.update({
                'status': 'identified',
                'confidence': match['confidence'],
                'card_name': card.get('printed_name'),
                'card_type': card.get('printed_typebox'),
                'card_id': card.get('card_id'),
                'print_id': card.get('print_id'),
                'pitch': card.get('printed_pitch'),
                'cost': card.get('printed_cost'),
                'power': card.get('printed_power'),
                'defense': card.get('printed_defense'),
                'artist': card.get('printed_artist'),
                'rules_text': card.get('printed_rules_text', '')[:200],  # First 200 chars
                'image_url': card.get('faces', [{}])[0].get('image', {}).get('normal', '') if card.get('faces') else ''
            })
        else:
            result['status'] = 'detected_but_not_identified'

        return result

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
        print("\n" + "="*60)
        print("SCAN COMPLETE")
        print("="*60)

        identification = result['identification']

        if identification['status'] == 'identified':
            print(f"\n✓ CARD IDENTIFIED (Confidence: {identification['confidence']:.2%})")
            print(f"\nName:       {identification['card_name']}")
            print(f"Type:       {identification['card_type']}")
            print(f"Card ID:    {identification['card_id']}")
            print(f"Print ID:   {identification['print_id']}")

            if identification.get('pitch'):
                print(f"Pitch:      {identification['pitch']}")
            if identification.get('cost'):
                print(f"Cost:       {identification['cost']}")
            if identification.get('power'):
                print(f"Power:      {identification['power']}")
            if identification.get('defense'):
                print(f"Defense:    {identification['defense']}")
            if identification.get('artist'):
                print(f"Artist:     {identification['artist']}")

            if identification.get('rules_text'):
                print(f"\nRules Text (excerpt):\n{identification['rules_text']}")

            if identification.get('image_url'):
                print(f"\nCard Image: {identification['image_url']}")

        else:
            print(f"\nStatus: {identification['status']}")
            if identification.get('detected_print_id'):
                print(f"Detected Print ID: {identification['detected_print_id']}")
            print(f"OCR extracted {len(identification.get('ocr_text', ''))} characters")
            print(f"Found {identification.get('api_results_count', 0)} API results")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

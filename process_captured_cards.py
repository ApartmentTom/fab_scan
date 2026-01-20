#!/usr/bin/env python3
"""
Batch process all cards captured with webcam system
"""

import os
import glob
from scan_card import CardScanner


def main():
    # Find all captured card images
    capture_dir = "captured_cards"

    if not os.path.exists(capture_dir):
        print(f"Error: Directory '{capture_dir}' not found")
        print("Run 'python webcam_capture.py' first to capture cards")
        return 1

    image_files = sorted(glob.glob(os.path.join(capture_dir, "*.png")))

    if not image_files:
        print(f"No images found in '{capture_dir}/'")
        print("Capture some cards first with: python webcam_capture.py")
        return 1

    print("="*60)
    print(f"PROCESSING CAPTURED CARDS")
    print("="*60)
    print(f"Found {len(image_files)} card images\n")

    scanner = CardScanner()
    output_dir = "captured_cards_processed"

    results = []

    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n{'='*60}")
        print(f"CARD {i}/{len(image_files)}: {filename}")
        print('='*60)

        try:
            result = scanner.scan(image_path, output_dir)

            # Display result
            ident = result['identification']

            if ident['status'] == 'identified':
                print(f"\n✓ IDENTIFIED: {ident['card_name']}")
                print(f"  Confidence: {ident['confidence']:.1%}")
                print(f"  Type: {ident['card_type']}")
                print(f"  Print ID: {ident.get('print_id', 'N/A')}")

                results.append({
                    'file': filename,
                    'success': True,
                    'identified': True,
                    'card': ident
                })
            else:
                print(f"\n✗ NOT IDENTIFIED")
                if ident.get('detected_print_id'):
                    print(f"  Detected Print ID: {ident['detected_print_id']} (no match)")
                print(f"  Status: {ident['status']}")

                results.append({
                    'file': filename,
                    'success': True,
                    'identified': False
                })

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            results.append({
                'file': filename,
                'success': False,
                'error': str(e)
            })

    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    successful = sum(1 for r in results if r['success'])
    identified = sum(1 for r in results if r.get('identified', False))

    print(f"Total cards: {len(results)}")
    print(f"Successfully processed: {successful}/{len(results)}")
    print(f"Successfully identified: {identified}/{successful}")
    print(f"Identification rate: {(identified/len(results)*100):.1f}%")

    if identified > 0:
        print(f"\n{'='*60}")
        print("IDENTIFIED CARDS")
        print('='*60)

        for r in results:
            if r.get('identified'):
                card = r['card']
                print(f"\n{r['file']}:")
                print(f"  → {card['card_name']}")
                print(f"     {card['card_type']}")
                print(f"     Confidence: {card['confidence']:.1%}")

    if output_dir and os.path.exists(output_dir):
        print(f"\n{'='*60}")
        print(f"Processed images saved to: {output_dir}/")

    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())

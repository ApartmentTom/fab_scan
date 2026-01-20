#!/usr/bin/env python3
"""
Batch test all card images in the repository
"""

import os
import glob
from scan_card import CardScanner


def main():
    # Find all PNG images in current directory
    image_files = glob.glob("*.png")

    if not image_files:
        print("No PNG images found in current directory")
        return

    print(f"Found {len(image_files)} images to scan\n")

    scanner = CardScanner()
    output_dir = "output"

    results = []

    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n{'='*60}")
        print(f"SCANNING IMAGE {i}/{len(image_files)}: {image_path}")
        print('='*60)

        try:
            result = scanner.scan(image_path, output_dir)
            results.append({
                'file': image_path,
                'success': True,
                'result': result
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'file': image_path,
                'success': False,
                'error': str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    successful = sum(1 for r in results if r['success'])
    print(f"Successfully processed: {successful}/{len(results)} images")

    if output_dir and os.path.exists(output_dir):
        print(f"\nProcessed images saved to: {output_dir}/")

    return 0


if __name__ == "__main__":
    exit(main())

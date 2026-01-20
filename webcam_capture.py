#!/usr/bin/env python3
"""
Webcam-based card capture system with real-time feedback and auto-alignment
"""

import cv2
import numpy as np
from datetime import datetime
import os
import sys


class WebcamCardCapture:
    """Live webcam capture system for card scanning"""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.output_dir = "captured_cards"
        self.min_card_area = 30000  # Minimum area for card detection
        self.capture_count = 0

        # Auto-capture settings
        self.auto_capture_enabled = True
        self.frames_stable = 0
        self.required_stable_frames = 15  # Need 15 consecutive good frames
        self.last_card_area = 0

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_camera(self):
        """Initialize webcam connection"""
        print(f"Opening camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # Get actual resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {width}x{height}")

        return True

    def detect_card_in_frame(self, frame):
        """Detect card boundaries in the frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find largest rectangular contour
        largest_rect = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > self.min_card_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Check aspect ratio (cards are roughly 1.4:1)
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 1.8:
                    max_area = area
                    largest_rect = {
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    }

        return largest_rect

    def check_card_quality(self, frame, card_rect):
        """Check if card is properly positioned and quality is good"""
        if not card_rect:
            return False, "No card detected"

        x, y, w, h = card_rect['bbox']
        frame_height, frame_width = frame.shape[:2]

        # Check if card is too small (too far from camera)
        card_width_percent = (w / frame_width) * 100
        if card_width_percent < 30:
            return False, f"Card too small ({card_width_percent:.0f}% of frame) - move closer"

        # Check if card is too large (too close to camera)
        if card_width_percent > 85:
            return False, f"Card too large ({card_width_percent:.0f}% of frame) - move further"

        # Check if card is centered
        card_center_x = x + w / 2
        card_center_y = y + h / 2
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        offset_x = abs(card_center_x - frame_center_x) / frame_width * 100
        offset_y = abs(card_center_y - frame_center_y) / frame_height * 100

        if offset_x > 15:
            direction = "right" if card_center_x < frame_center_x else "left"
            return False, f"Card off-center - move {direction}"

        if offset_y > 15:
            direction = "down" if card_center_y < frame_center_y else "up"
            return False, f"Card off-center - move {direction}"

        # Check aspect ratio
        aspect = card_rect['aspect_ratio']
        if not (1.35 < aspect < 1.55):
            return False, f"Card tilted (aspect: {aspect:.2f}) - straighten card"

        # Check for motion blur (variance of Laplacian)
        card_roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < 50:
            return False, f"Image blurry (focus: {laplacian_var:.0f}) - hold still"

        return True, f"Perfect! ({card_width_percent:.0f}% frame, focus: {laplacian_var:.0f})"

    def draw_feedback_overlay(self, frame, card_rect, quality_ok, message):
        """Draw visual feedback on the frame"""
        overlay = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        # Draw alignment guides (center crosshair)
        center_x, center_y = frame_width // 2, frame_height // 2
        color = (0, 255, 0) if quality_ok else (0, 165, 255)
        cv2.line(overlay, (center_x - 30, center_y), (center_x + 30, center_y), color, 2)
        cv2.line(overlay, (center_x, center_y - 30), (center_x, center_y + 30), color, 2)

        # Draw target zone (where card should be)
        target_w = int(frame_width * 0.5)
        target_h = int(target_w * 1.4)
        target_x = (frame_width - target_w) // 2
        target_y = (frame_height - target_h) // 2
        cv2.rectangle(overlay, (target_x, target_y),
                     (target_x + target_w, target_y + target_h),
                     (100, 100, 100), 2)

        # Draw detected card rectangle
        if card_rect:
            x, y, w, h = card_rect['bbox']
            rect_color = (0, 255, 0) if quality_ok else (0, 165, 255)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), rect_color, 3)

            # Draw corners
            corner_len = 20
            for corner_x, corner_y in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
                cv2.circle(overlay, (corner_x, corner_y), 5, rect_color, -1)

        # Draw status message
        if quality_ok:
            status_text = f"READY TO CAPTURE! {self.frames_stable}/{self.required_stable_frames}"
            status_color = (0, 255, 0)
        else:
            status_text = message
            status_color = (0, 165, 255)

        # Background for text
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(overlay, (10, 10), (20 + text_size[0], 50), (0, 0, 0), -1)
        cv2.putText(overlay, status_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Draw instructions
        instructions = [
            "SPACEBAR: Manual capture",
            "A: Toggle auto-capture" + (" [ON]" if self.auto_capture_enabled else " [OFF]"),
            "Q: Quit",
        ]

        y_offset = frame_height - 100
        for i, instruction in enumerate(instructions):
            cv2.putText(overlay, instruction, (15, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend overlay
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def save_capture(self, frame, card_rect):
        """Save captured card image"""
        if card_rect:
            # Crop to card
            x, y, w, h = card_rect['bbox']
            # Add small padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            cropped = frame[y1:y2, x1:x2]
        else:
            cropped = frame

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"card_{self.capture_count:04d}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        # Save image
        cv2.imwrite(filepath, cropped)
        self.capture_count += 1

        print(f"\n✓ Captured: {filename}")
        return filepath

    def run(self):
        """Main capture loop"""
        if not self.initialize_camera():
            return

        print("\n" + "="*60)
        print("WEBCAM CARD CAPTURE")
        print("="*60)
        print("\nControls:")
        print("  SPACEBAR - Manual capture")
        print("  A        - Toggle auto-capture")
        print("  Q        - Quit")
        print(f"\nAuto-capture: {'ENABLED' if self.auto_capture_enabled else 'DISABLED'}")
        print(f"Images will be saved to: {self.output_dir}/")
        print("\nPositioning your card:")
        print("  • Place card flat in center of frame")
        print("  • Align with target rectangle")
        print("  • Hold still when 'READY' appears")
        print("="*60 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Detect card
                card_rect = self.detect_card_in_frame(frame)

                # Check quality
                quality_ok, message = self.check_card_quality(frame, card_rect)

                # Auto-capture logic
                if quality_ok and card_rect:
                    # Check stability
                    current_area = card_rect['area']
                    area_diff = abs(current_area - self.last_card_area) / current_area if current_area > 0 else 1.0

                    if area_diff < 0.02:  # Less than 2% change
                        self.frames_stable += 1
                    else:
                        self.frames_stable = 0

                    self.last_card_area = current_area

                    # Auto-capture when stable
                    if self.auto_capture_enabled and self.frames_stable >= self.required_stable_frames:
                        self.save_capture(frame, card_rect)
                        self.frames_stable = 0
                        # Brief pause after capture
                        cv2.waitKey(1000)
                else:
                    self.frames_stable = 0

                # Draw feedback
                display_frame = self.draw_feedback_overlay(
                    frame.copy(), card_rect, quality_ok, message
                )

                # Show frame
                cv2.imshow('Card Capture', display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # Spacebar - manual capture
                    filepath = self.save_capture(frame, card_rect)
                    # Brief flash feedback
                    white_frame = np.ones_like(frame) * 255
                    cv2.imshow('Card Capture', white_frame)
                    cv2.waitKey(100)
                elif key == ord('a'):  # Toggle auto-capture
                    self.auto_capture_enabled = not self.auto_capture_enabled
                    status = "ENABLED" if self.auto_capture_enabled else "DISABLED"
                    print(f"\nAuto-capture: {status}")
                    self.frames_stable = 0

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            self.cleanup()

        print(f"\n{'='*60}")
        print(f"Session complete!")
        print(f"Captured {self.capture_count} cards")
        print(f"Saved to: {self.output_dir}/")
        print("="*60 + "\n")

    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    # Parse camera index from command line
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index: {sys.argv[1]}")
            print("Usage: python webcam_capture.py [camera_index]")
            return 1

    # Create and run capture system
    try:
        capture = WebcamCardCapture(camera_index=camera_index)
        capture.run()
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  • Check that your webcam is connected")
        print("  • Close other applications using the webcam")
        print("  • Try a different camera index: python webcam_capture.py 1")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

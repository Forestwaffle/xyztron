#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class TrafficLightDetector:
    def __init__(self, show_debug=True):
        # Detector activation flag
        self.active = False

        # Current result
        self.state = "STOP"
        self.detected_light = "UNKNOWN"

        # Consecutive frame counters
        self.red_count = 0
        self.green_count = 0

        # Debug option
        self.show_debug = show_debug

        # Crop ratio for upper camera area
        self.crop_y_ratio = 0.42

        # Color thresholds
        self.red_threshold = 0.020
        self.green_threshold = 0.030

        # Required consecutive frames
        self.red_confirm_count = 2
        self.green_confirm_count = 3

    def enable(self):
        """Enable detector only once."""
        if self.active:
            return

        self.active = True
        self.reset()

    def disable(self):
        """Disable detector."""
        self.active = False
        self.reset()

    def reset(self):
        """Reset internal state."""
        self.state = "STOP"
        self.detected_light = "UNKNOWN"
        self.red_count = 0
        self.green_count = 0

    def is_go(self):
        """Return True if state is GO."""
        return self.state == "GO"

    def is_stop(self):
        """Return True if state is STOP."""
        return self.state == "STOP"

    def process(self, frame):
        """
        Process one camera frame.

        Returns:
            state: "STOP" or "GO"
            detected_light: "RED", "GREEN", "UNKNOWN", or "DISABLED"
            debug_frame: cropped camera frame
        """
        if frame is None:
            return self.state, self.detected_light, None

        height, width = frame.shape[:2]

        crop_y1 = 0
        crop_y2 = int(height * self.crop_y_ratio)
        crop_x1 = 0
        crop_x2 = width

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # =====================================================
        # Detector disabled:
        # Do not analyze traffic light,
        # but keep OpenCV debug window updating.
        # =====================================================
        if not self.active:
            if self.show_debug:
                cv2.imshow("Traffic Light Detector", cropped)
                cv2.waitKey(1)

            return self.state, "DISABLED", cropped

        traffic_box = self.find_black_traffic_box(cropped)

        self.detected_light = "UNKNOWN"

        if traffic_box is not None:
            self.detected_light = self.analyze_red_green_inside_box(
                cropped,
                traffic_box
            )
        else:
            self.red_count = 0
            self.green_count = 0

        if self.detected_light == "RED":
            self.red_count += 1
            self.green_count = 0

        elif self.detected_light == "GREEN":
            self.green_count += 1
            self.red_count = 0

        else:
            self.red_count = 0
            self.green_count = 0

        # Red has priority
        if self.red_count >= self.red_confirm_count:
            self.state = "STOP"

        if self.green_count >= self.green_confirm_count:
            self.state = "GO"

        # Show debug window
        if self.show_debug:
            cv2.imshow("Traffic Light Detector", cropped)
            cv2.waitKey(1)

        return self.state, self.detected_light, cropped

    def find_black_traffic_box(self, image):
        """Find the black rectangular traffic light box."""
        height, width = image.shape[:2]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        black_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 0]),
            np.array([180, 130, 100])
        )

        close_kernel = np.ones((11, 11), np.uint8)
        open_kernel = np.ones((5, 5), np.uint8)

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_OPEN,
            open_kernel
        )

        contours, _ = cv2.findContours(
            black_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_box = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 700:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w < 80 or h < 25:
                continue

            aspect_ratio = w / float(h)
            rect_area = w * h
            fill_ratio = area / float(rect_area)

            if aspect_ratio < 1.7 or aspect_ratio > 6.0:
                continue

            if fill_ratio < 0.25:
                continue

            center_y = y + h / 2.0

            if center_y > height * 0.85:
                continue

            score = area + rect_area

            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)

        if best_box is not None:
            x, y, w, h = best_box

            # Draw detected traffic light box only.
            # No text is drawn on the screen.
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (255, 255, 0),
                3
            )

        return best_box

    def analyze_red_green_inside_box(self, image, box):
        """Check red and green pixel ratios inside the detected box."""
        x, y, w, h = box

        pad_x = int(w * 0.08)
        pad_y = int(h * 0.12)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, image.shape[1])
        y2 = min(y + h + pad_y, image.shape[0])

        panel = image[y1:y2, x1:x2]

        if panel.size == 0:
            return "UNKNOWN"

        hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)

        red_mask_1 = cv2.inRange(
            hsv,
            np.array([0, 100, 100]),
            np.array([10, 255, 255])
        )

        red_mask_2 = cv2.inRange(
            hsv,
            np.array([170, 100, 100]),
            np.array([180, 255, 255])
        )

        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        green_mask = cv2.inRange(
            hsv,
            np.array([45, 120, 120]),
            np.array([85, 255, 255])
        )

        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        total_pixels = panel.shape[0] * panel.shape[1]

        red_ratio = red_pixels / float(total_pixels)
        green_ratio = green_pixels / float(total_pixels)

        # Draw analysis area only.
        # Removed R/G ratio text from the debug window.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        # Red priority rule
        if red_ratio > self.red_threshold:
            return "RED"

        if green_ratio > self.green_threshold:
            return "GREEN"

        return "UNKNOWN"
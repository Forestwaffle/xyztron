#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 오른쪽 흰색 차선 거리 유지 주행 로직.

    동작:
        1. 전방 카메라 이미지에서 아래쪽 + 오른쪽 ROI만 사용
        2. 흰색 차선만 검출
        3. HoughLinesP로 오른쪽 차선 선분 검출
        4. 오른쪽 차선의 특정 y 위치에서 x 좌표를 계산
        5. 오른쪽 차선 x 좌표가 목표 x 위치에 오도록 조향
        6. 차선이 보이면 빠르게 주행, 안 보이면 오른쪽으로 느리게 탐색 주행

    출력:
        angle: -100.0 ~ 100.0
        speed:
            흰색 차선 보임    -> 16.0
            흰색 차선 안 보임 -> 8.0
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE Right Lane Distance"

        # =====================================================
        # Drive parameters
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        self.line_detected_speed = 16.0
        self.no_line_speed = 8.0
        self.max_angle = 100.0

        # 차선을 못 찾았을 때 탐색 조향각.
        # 오른쪽으로 가야 하는데 왼쪽으로 가면 부호를 반대로 바꾸면 됨.
        self.search_angle = 20.0

        # =====================================================
        # Right-lane distance control parameters
        # =====================================================
        # 오른쪽 흰색 선을 화면의 몇 % 지점에 유지할지 결정.
        # 값이 클수록 오른쪽 선을 화면 오른쪽에 두려고 함.
        # 일반 추천: 0.72 ~ 0.85
        self.target_right_line_x_ratio = 0.82

        # 차선 x 위치를 계산할 기준 y 위치.
        # 0.85면 화면 아래쪽 85% 높이에서 차선 x를 계산.
        self.lookahead_y_ratio = 0.85

        # 거리 오차 기반 조향 민감도.
        # pixel error를 조향각으로 바꾸는 gain.
        self.distance_gain = 0.18

        # 기울기 보조 제어.
        # 0.0으로 두면 거리만 보고 조향.
        # 너무 휘청이면 0.2~0.5 정도 사용.
        self.heading_gain = 0.25
        self.target_line_angle_deg = 55.0

        # 전체 조향 방향 보정값.
        # 차가 반대로 꺾이면 -1.0을 1.0으로 바꾸면 됨.
        self.steering_sign = -1.0

        # 조향 smoothing.
        self.smoothing_alpha = 0.60
        self.prev_angle = 0.0

        # =====================================================
        # ROI parameters
        # =====================================================
        self.roi_y_start_ratio = 0.50
        self.roi_x_start_ratio = 0.40

        # =====================================================
        # White color threshold
        # =====================================================
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 70, 255])
        self.min_bgr_white = 160

        # =====================================================
        # Edge / Hough parameters
        # =====================================================
        self.canny_low = 50
        self.canny_high = 150

        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 25
        self.hough_min_line_length = 25
        self.hough_max_line_gap = 25

        self.min_abs_slope = 0.30
        self.max_abs_slope = 6.0
        self.min_line_length = 25.0

        # 오른쪽 차선은 이미지에서 보통 "/" 형태라 양수 기울기.
        # 실제 화면에서 오른쪽 차선이 반대로 잡히면 False로 변경.
        self.use_positive_slope_only = True

        # =====================================================
        # State / debug
        # =====================================================
        self.started = False
        self.warned_no_image = False
        self.log_count = 0

        self.lane_detected = False
        self.last_line_count = 0
        self.last_line_angle_deg = None
        self.last_line_x = None
        self.last_target_x = None
        self.last_distance_error = None
        self.last_debug_data = None

    # =====================================================
    # Lifecycle
    # =====================================================

    def start(self):
        if self.started:
            return

        self.started = True
        self.log_info(
            "AutoDrive started: right white lane distance control "
            f"| target_x_ratio:{self.target_right_line_x_ratio:.2f} "
            f"| fast:{self.line_detected_speed:.1f} "
            f"| slow:{self.no_line_speed:.1f}"
        )

    def stop(self):
        self.angle = 0.0
        self.speed = 0.0
        self.prev_angle = 0.0
        self.started = False

        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

        self.log_info("AutoDrive stopped")

    # =====================================================
    # Main process
    # =====================================================

    def process(self, image):
        if image is None:
            if not self.warned_no_image:
                self.log_warn("AUTO_DRIVE: waiting for camera image...")
                self.warned_no_image = True

            self.angle = self.search_angle
            self.speed = self.no_line_speed
            return self.angle, self.speed

        self.warned_no_image = False

        result = self.detect_right_white_lane(image)

        if result is None:
            # 흰색 오른쪽 차선이 안 보이면 오른쪽으로 느리게 탐색 주행
            self.lane_detected = False
            self.last_line_count = 0
            self.last_line_angle_deg = None
            self.last_line_x = None
            self.last_target_x = None
            self.last_distance_error = None

            self.angle = self.search_angle
            self.speed = self.no_line_speed
            self.prev_angle = self.angle

        else:
            lane_x, target_x, line_angle_deg, line_count, debug_data = result

            self.lane_detected = True
            self.last_line_x = lane_x
            self.last_target_x = target_x
            self.last_line_angle_deg = line_angle_deg
            self.last_line_count = line_count

            # 거리 오차: 오른쪽 선이 목표 위치보다 왼쪽/오른쪽에 있는 정도
            distance_error = target_x - lane_x
            self.last_distance_error = distance_error

            # 기울기 오차는 보조로만 사용
            heading_error = self.target_line_angle_deg - line_angle_deg

            raw_angle = (
                self.steering_sign
                * (
                    self.distance_gain * distance_error
                    + self.heading_gain * heading_error
                )
            )

            raw_angle = self.clamp(raw_angle, -self.max_angle, self.max_angle)

            smoothed_angle = (
                self.smoothing_alpha * self.prev_angle
                + (1.0 - self.smoothing_alpha) * raw_angle
            )

            self.angle = self.clamp(
                smoothed_angle,
                -self.max_angle,
                self.max_angle
            )
            self.speed = self.line_detected_speed
            self.prev_angle = self.angle

        if self.show_debug:
            self.show_debug_view(image)

        self.print_log()

        return self.angle, self.speed

    # =====================================================
    # Lane detection
    # =====================================================

    def detect_right_white_lane(self, image):
        height, width = image.shape[:2]

        roi_y1 = int(height * self.roi_y_start_ratio)
        roi_y2 = height
        roi_x1 = int(width * self.roi_x_start_ratio)
        roi_x2 = width

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()

        if roi.size == 0:
            return None

        white_mask = self.create_white_mask(roi)

        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(white_mask, self.canny_low, self.canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        if lines is None:
            return None

        valid_lines = []

        lookahead_y_full = int(height * self.lookahead_y_ratio)
        lookahead_y_roi = lookahead_y_full - roi_y1
        target_x_full = float(width * self.target_right_line_x_ratio)

        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            length = float(np.hypot(dx, dy))

            if length < self.min_line_length:
                continue

            slope = dy / float(dx)
            abs_slope = abs(slope)

            if abs_slope < self.min_abs_slope:
                continue

            if abs_slope > self.max_abs_slope:
                continue

            if self.use_positive_slope_only and slope <= 0:
                continue

            # lookahead_y 위치에서 이 선분이 가지는 x 좌표 계산
            # x = x1 + (y - y1) / slope
            lane_x_roi = x1 + (lookahead_y_roi - y1) / slope
            lane_x_full = lane_x_roi + roi_x1

            # ROI 밖으로 심하게 벗어난 추정값 제거
            if lane_x_full < 0 or lane_x_full > width:
                continue

            angle_deg = abs(float(np.degrees(np.arctan2(dy, dx))))

            valid_lines.append({
                "line": (x1, y1, x2, y2),
                "slope": slope,
                "angle_deg": angle_deg,
                "length": length,
                "lane_x_full": float(lane_x_full)
            })

        if len(valid_lines) == 0:
            return None

        total_weight = sum(item["length"] for item in valid_lines)

        if total_weight <= 0.0:
            return None

        weighted_x = 0.0
        weighted_angle = 0.0

        for item in valid_lines:
            weighted_x += item["lane_x_full"] * item["length"]
            weighted_angle += item["angle_deg"] * item["length"]

        lane_x_full = weighted_x / total_weight
        line_angle_deg = weighted_angle / total_weight

        debug_data = {
            "roi": roi,
            "white_mask": white_mask,
            "edges": edges,
            "valid_lines": valid_lines,
            "roi_offset": (roi_x1, roi_y1),
            "lookahead_y_full": lookahead_y_full,
            "target_x_full": target_x_full,
            "lane_x_full": lane_x_full
        }

        self.last_debug_data = debug_data

        return (
            float(lane_x_full),
            float(target_x_full),
            float(line_angle_deg),
            len(valid_lines),
            debug_data
        )

    def create_white_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_white_mask = cv2.inRange(
            hsv,
            self.white_lower,
            self.white_upper
        )

        b, g, r = cv2.split(image)
        min_channel = cv2.min(cv2.min(b, g), r)

        bgr_white_mask = cv2.inRange(
            min_channel,
            self.min_bgr_white,
            255
        )

        white_mask = cv2.bitwise_and(hsv_white_mask, bgr_white_mask)

        return white_mask

    # =====================================================
    # Debug view
    # =====================================================

    def show_debug_view(self, image):
        debug_frame = image.copy()
        height, width = debug_frame.shape[:2]

        roi_y1 = int(height * self.roi_y_start_ratio)
        roi_y2 = height
        roi_x1 = int(width * self.roi_x_start_ratio)
        roi_x2 = width

        cv2.rectangle(
            debug_frame,
            (roi_x1, roi_y1),
            (roi_x2, roi_y2),
            (255, 255, 0),
            2
        )

        lookahead_y = int(height * self.lookahead_y_ratio)

        cv2.line(
            debug_frame,
            (0, lookahead_y),
            (width, lookahead_y),
            (255, 0, 255),
            2
        )

        if self.last_target_x is not None:
            target_x = int(self.last_target_x)

            cv2.line(
                debug_frame,
                (target_x, roi_y1),
                (target_x, height),
                (255, 255, 255),
                2
            )

            cv2.circle(
                debug_frame,
                (target_x, lookahead_y),
                8,
                (255, 255, 255),
                -1
            )

        if self.last_line_x is not None:
            lane_x = int(self.last_line_x)

            cv2.line(
                debug_frame,
                (lane_x, roi_y1),
                (lane_x, height),
                (0, 0, 255),
                2
            )

            cv2.circle(
                debug_frame,
                (lane_x, lookahead_y),
                8,
                (0, 0, 255),
                -1
            )

        if self.last_debug_data is not None and self.lane_detected:
            valid_lines = self.last_debug_data.get("valid_lines", [])
            offset_x, offset_y = self.last_debug_data.get("roi_offset", (0, 0))

            for item in valid_lines:
                x1, y1, x2, y2 = item["line"]

                cv2.line(
                    debug_frame,
                    (x1 + offset_x, y1 + offset_y),
                    (x2 + offset_x, y2 + offset_y),
                    (0, 255, 0),
                    3
                )

        status_text = (
            "RIGHT LANE DETECTED"
            if self.lane_detected
            else "NO WHITE RIGHT LANE"
        )
        status_color = (0, 255, 0) if self.lane_detected else (0, 0, 255)

        cv2.putText(
            debug_frame,
            "MISSION: AUTO_DRIVE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            status_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            status_color,
            2
        )

        cv2.putText(
            debug_frame,
            f"ANGLE:{self.angle:.2f} SPEED:{self.speed:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )

        if self.last_distance_error is not None:
            cv2.putText(
                debug_frame,
                f"LANE_X:{self.last_line_x:.1f} "
                f"TARGET_X:{self.last_target_x:.1f} "
                f"ERR:{self.last_distance_error:.1f}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                debug_frame,
                "LANE_X:NONE TARGET_X:NONE",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

        if self.last_line_angle_deg is not None:
            cv2.putText(
                debug_frame,
                f"LINE_ANGLE:{self.last_line_angle_deg:.1f} "
                f"COUNT:{self.last_line_count}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

        cv2.imshow(self.window_name, debug_frame)
        cv2.waitKey(1)

    # =====================================================
    # Utility
    # =====================================================

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def print_log(self):
        self.log_count += 1

        if self.log_count % 20 != 0:
            return

        lane_x_text = (
            "None"
            if self.last_line_x is None
            else f"{self.last_line_x:.1f}"
        )

        target_x_text = (
            "None"
            if self.last_target_x is None
            else f"{self.last_target_x:.1f}"
        )

        err_text = (
            "None"
            if self.last_distance_error is None
            else f"{self.last_distance_error:.1f}"
        )

        line_angle_text = (
            "None"
            if self.last_line_angle_deg is None
            else f"{self.last_line_angle_deg:.1f}"
        )

        self.log_info(
            f"AUTO_DRIVE | "
            f"lane:{self.lane_detected} | "
            f"lane_x:{lane_x_text} | "
            f"target_x:{target_x_text} | "
            f"err:{err_text} | "
            f"line_angle:{line_angle_text} | "
            f"count:{self.last_line_count} | "
            f"angle:{self.angle:.2f} | "
            f"speed:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)
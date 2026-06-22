#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 오른쪽 흰색 차선 기반 주행 로직.

    동작:
        1. 전방 카메라 이미지에서 오른쪽 영역만 사용
        2. 흰색 차선만 검출
        3. HoughLinesP로 오른쪽 차선 선분 검출
        4. 선분 기울기를 목표 기울기에 맞춰 조향
        5. 흰색 차선이 보이면 빠르게 주행
        6. 흰색 차선이 안 보이면 느린 속도로 직진

    출력:
        angle: -100.0 ~ 100.0
        speed:
            흰색 차선 보임    -> 16.0
            흰색 차선 안 보임 -> 8.0
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE Right White Lane"

        # =====================================================
        # Drive parameters
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        # 흰색 오른쪽 차선이 보일 때 속도
        self.line_detected_speed = 8.0

        # 흰색 차선이 안 보일 때 느린 직진 속도
        self.no_line_speed = 8.0

        # 최대 조향각
        self.max_angle = 100.0

        # =====================================================
        # Steering parameters
        # =====================================================
        # 오른쪽 차선의 목표 기울기 각도.
        # 보통 오른쪽 차선은 이미지에서 / 방향이므로 50~70도 근처가 나옴.
        self.target_line_angle_deg = 55.0

        # 조향 민감도
        self.steering_gain = 1.4

        # 조향 방향 보정값.
        # 차가 반대로 꺾이면 -1.0을 1.0으로 바꾸면 됨.
        self.steering_sign = -1.0

        # 조향 smoothing.
        # 0.0이면 smoothing 없음.
        # 0.7이면 이전 조향을 70%, 새 조향을 30% 반영.
        self.smoothing_alpha = 0.65

        self.prev_angle = 0.0

        # =====================================================
        # ROI parameters
        # =====================================================
        # 화면 아래쪽만 사용
        self.roi_y_start_ratio = 0.52

        # 화면 오른쪽만 사용
        self.roi_x_start_ratio = 0.45

        # =====================================================
        # White color threshold
        # =====================================================
        # HSV 기준 흰색:
        # H는 전체 허용, S는 낮고, V는 높은 픽셀
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 70, 255])

        # BGR 각 채널 최소 밝기 조건
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

        # 유효 선분 조건
        self.min_abs_slope = 0.35
        self.max_abs_slope = 5.0
        self.min_line_length = 25.0

        # 오른쪽 차선은 일반적으로 양수 기울기
        # 이미지 좌표계에서 x 오른쪽, y 아래쪽 기준:
        # 오른쪽 차선은 대체로 "/" 형태 -> slope 양수
        self.use_positive_slope_only = True

        # =====================================================
        # State / logging
        # =====================================================
        self.started = False
        self.warned_no_image = False
        self.log_count = 0

        self.lane_detected = False
        self.last_line_angle_deg = None
        self.last_line_count = 0

    # =====================================================
    # Lifecycle
    # =====================================================

    def start(self):
        if self.started:
            return

        self.started = True
        self.log_info(
            "AutoDrive started: right white lane slope control "
            f"| fast_speed:{self.line_detected_speed:.1f} "
            f"| slow_speed:{self.no_line_speed:.1f}"
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
        """
        AUTO_DRIVE 메인 로직.

        Args:
            image: OpenCV BGR image

        Returns:
            angle, speed
        """
        if image is None:
            if not self.warned_no_image:
                self.log_warn("AUTO_DRIVE: waiting for camera image...")
                self.warned_no_image = True

            # 카메라가 없으면 안전하게 느린 직진
            self.angle = 0.0
            self.speed = self.no_line_speed
            return self.angle, self.speed

        self.warned_no_image = False

        result = self.detect_right_white_lane(image)

        if result is None:
            # 흰색 차선이 안 보이면 느린 속도로 직진
            self.lane_detected = False
            self.last_line_angle_deg = None
            self.last_line_count = 0

            self.angle = 20.0
            self.speed = self.no_line_speed

            # 차선이 안 보일 때는 이전 조향값 제거
            self.prev_angle = self.angle

        else:
            line_angle_deg, line_count, debug_data = result

            self.lane_detected = True
            self.last_line_angle_deg = line_angle_deg
            self.last_line_count = line_count

            # 목표 기울기와 현재 기울기의 차이
            angle_error = self.target_line_angle_deg - line_angle_deg

            raw_angle = (
                self.steering_sign
                * self.steering_gain
                * angle_error
            )

            raw_angle = self.clamp(
                raw_angle,
                -self.max_angle,
                self.max_angle
            )

            # smoothing 적용
            smoothed_angle = (
                self.smoothing_alpha * self.prev_angle
                + (1.0 - self.smoothing_alpha) * raw_angle
            )

            smoothed_angle = self.clamp(
                smoothed_angle,
                -self.max_angle,
                self.max_angle
            )

            self.angle = smoothed_angle
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
        """
        오른쪽 ROI에서 흰색 차선 선분을 찾고 평균 기울기 각도를 반환.

        Returns:
            None
            or
            (line_angle_deg, line_count, debug_data)
        """
        height, width = image.shape[:2]

        # ROI 설정
        roi_y1 = int(height * self.roi_y_start_ratio)
        roi_y2 = height
        roi_x1 = int(width * self.roi_x_start_ratio)
        roi_x2 = width

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()

        if roi.size == 0:
            return None

        # 흰색 마스크 생성
        white_mask = self.create_white_mask(roi)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            kernel
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Edge 검출
        edges = cv2.Canny(
            white_mask,
            self.canny_low,
            self.canny_high
        )

        # Hough line 검출
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

            # 오른쪽 차선만 사용: 양수 기울기 선분
            if self.use_positive_slope_only and slope <= 0:
                continue

            angle_deg = np.degrees(np.arctan2(dy, dx))

            # 각도는 절댓값 기준으로 사용
            # 오른쪽 차선이면 보통 +각도
            angle_deg = abs(float(angle_deg))

            valid_lines.append({
                "line": (x1, y1, x2, y2),
                "slope": slope,
                "angle_deg": angle_deg,
                "length": length
            })

        if len(valid_lines) == 0:
            return None

        # 긴 선분일수록 더 크게 반영
        total_weight = sum(item["length"] for item in valid_lines)

        if total_weight <= 0.0:
            return None

        weighted_angle = 0.0

        for item in valid_lines:
            weighted_angle += item["angle_deg"] * item["length"]

        line_angle_deg = weighted_angle / total_weight

        debug_data = {
            "roi": roi,
            "white_mask": white_mask,
            "edges": edges,
            "valid_lines": valid_lines,
            "roi_offset": (roi_x1, roi_y1)
        }

        self.last_debug_data = debug_data

        return line_angle_deg, len(valid_lines), debug_data

    def create_white_mask(self, image):
        """
        흰색 차선만 검출하는 마스크 생성.
        노란색은 포함하지 않음.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_white_mask = cv2.inRange(
            hsv,
            self.white_lower,
            self.white_upper
        )

        # BGR 각 채널이 모두 충분히 밝은 픽셀만 추가 허용
        b, g, r = cv2.split(image)

        bgr_white_mask = cv2.inRange(
            cv2.min(cv2.min(b, g), r),
            self.min_bgr_white,
            255
        )

        white_mask = cv2.bitwise_and(
            hsv_white_mask,
            bgr_white_mask
        )

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

        # ROI 표시
        cv2.rectangle(
            debug_frame,
            (roi_x1, roi_y1),
            (roi_x2, roi_y2),
            (255, 255, 0),
            2
        )

        # 검출된 선분 표시
        if hasattr(self, "last_debug_data") and self.lane_detected:
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

        # 상태 텍스트
        status_text = "LANE DETECTED" if self.lane_detected else "NO WHITE LANE"
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
            0.85,
            status_color,
            2
        )

        cv2.putText(
            debug_frame,
            f"ANGLE: {self.angle:.2f}  SPEED: {self.speed:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        if self.last_line_angle_deg is not None:
            cv2.putText(
                debug_frame,
                f"LINE_ANGLE: {self.last_line_angle_deg:.2f} deg  "
                f"COUNT: {self.last_line_count}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                debug_frame,
                "LINE_ANGLE: NONE",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.putText(
            debug_frame,
            f"FAST:{self.line_detected_speed:.1f}  "
            f"SLOW:{self.no_line_speed:.1f}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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

        if self.last_line_angle_deg is None:
            line_angle_text = "None"
        else:
            line_angle_text = f"{self.last_line_angle_deg:.2f}"

        self.log_info(
            f"AUTO_DRIVE | "
            f"lane:{self.lane_detected} | "
            f"line_angle:{line_angle_text} | "
            f"line_count:{self.last_line_count} | "
            f"angle:{self.angle:.2f} | "
            f"speed:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 흰색 차선 중앙 추종 로직.

    목표:
        주변의 왼쪽/오른쪽 흰색 차선을 기준으로 차선 중앙을 계산하고,
        차량 중심이 그 중앙으로 가도록 조향한다.

    방식:
        1. 전방 카메라 이미지를 Bird's Eye View로 변환
        2. 아래쪽 ROI 사용
        3. 흰색 차선만 HSV + BGR 밝기 조건으로 검출
        4. 화면 중앙 기준 왼쪽/오른쪽 흰선 픽셀 분리
        5. 왼쪽/오른쪽 흰선을 각각 2차 곡선으로 피팅
        6. lookahead_y 위치에서 left_x, right_x 계산
        7. lane_center = (left_x + right_x) / 2
        8. 차량 중심과 lane_center 오차로 조향
        9. 한쪽 선이 안 보이면 이전 차선 폭으로 반대쪽 예측
        10. 둘 다 안 보이면 짧게 마지막 조향 유지 후 서서히 직진 복귀
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE White Lane Center"

        # =====================================================
        # Drive parameters
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        # 처음 테스트는 천천히
        self.line_detected_speed = 6.0
        self.prediction_speed = 4.5
        self.no_line_speed = 3.5

        self.max_angle = 100.0

        # =====================================================
        # Control parameters
        # =====================================================
        self.target_center_x_ratio = 0.50

        # 동영상 분석 기준: 너무 아래만 보면 커브 반응이 늦어서 0.45 사용
        self.roi_y_start_ratio = 0.45

        # 작을수록 더 앞쪽을 봄
        self.lookahead_y_ratio = 0.70
        self.near_y_ratio = 0.95

        # 중심선의 방향성 보정
        self.heading_gain = 0.35

        # 조향 방향 보정
        # 차가 반대로 꺾이면 -1.0을 1.0으로 바꿀 것
        self.steering_sign = -1.0

        # =====================================================
        # Dynamic response parameters
        # =====================================================
        # 반응 빠르게 + 오차 크면 강하게
        self.near_error_px = 10.0
        self.far_error_px = 70.0

        self.min_center_gain = 0.22
        self.max_center_gain = 0.85

        self.dead_zone_px = 3.0

        self.near_smoothing_alpha = 0.50
        self.far_smoothing_alpha = 0.12

        self.last_dynamic_gain = 0.0
        self.last_dynamic_smoothing = 0.0

        self.prev_angle = 0.0

        # =====================================================
        # White lane threshold
        # =====================================================
        # 흰색 HSV 조건: 채도 낮고 밝기 높은 픽셀
        self.white_lower = np.array([0, 0, 165])
        self.white_upper = np.array([180, 85, 255])

        # BGR 각 채널 최소 밝기
        self.min_bgr_white = 145

        # 픽셀 개수 조건
        self.min_total_white_pixels = 120
        self.min_side_pixels = 50

        # =====================================================
        # Lane fit / prediction parameters
        # =====================================================
        self.last_left_fit = None
        self.last_right_fit = None

        # 새 피팅 반영 비율.
        # 0.45면 이전 45%, 새 검출 55%
        self.fit_alpha = 0.45

        self.left_prediction_count = 0
        self.right_prediction_count = 0
        self.max_side_prediction_frames = 8

        # 차선 폭 추정
        self.default_lane_width_ratio = 0.52
        self.estimated_lane_width_px = None
        self.lane_width_alpha = 0.80

        self.min_lane_width_ratio = 0.25
        self.max_lane_width_ratio = 0.90

        # 둘 다 안 보일 때 마지막 조향 유지 제한
        self.hold_angle_count = 0
        self.max_hold_angle_frames = 6
        self.hold_angle_decay = 0.70

        # =====================================================
        # Debug / state
        # =====================================================
        self.started = False
        self.warned_no_image = False
        self.log_count = 0

        self.lane_available = False
        self.left_available = False
        self.right_available = False
        self.using_prediction = False
        self.holding_angle = False

        self.last_left_x = None
        self.last_right_x = None
        self.last_center_x = None
        self.last_center_near_x = None
        self.last_target_x = None

        self.last_center_error = None
        self.last_heading_error = None
        self.last_control_error = None

        self.last_left_count = 0
        self.last_right_count = 0
        self.last_lane_width = None

        self.last_warped = None
        self.last_mask = None
        self.last_roi_y1 = None

    # =====================================================
    # Lifecycle
    # =====================================================

    def start(self):
        if self.started:
            return

        self.started = True
        self.log_info(
            "AutoDrive started: white lane center tracking "
            f"| line_speed:{self.line_detected_speed:.1f} "
            f"| prediction_speed:{self.prediction_speed:.1f} "
            f"| no_line_speed:{self.no_line_speed:.1f}"
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

            self.angle = self.prev_angle
            self.speed = self.no_line_speed
            return self.angle, self.speed

        self.warned_no_image = False

        result = self.detect_white_lane_center(image)

        if result is None:
            # 둘 다 안 보이면 짧게 마지막 각도 유지 후 서서히 직진 복귀
            self.lane_available = False
            self.left_available = False
            self.right_available = False
            self.using_prediction = False
            self.holding_angle = True

            self.last_left_x = None
            self.last_right_x = None
            self.last_center_x = None
            self.last_center_near_x = None
            self.last_target_x = None

            self.last_center_error = None
            self.last_heading_error = None
            self.last_control_error = None
            self.last_dynamic_gain = 0.0
            self.last_dynamic_smoothing = 0.0

            self.hold_angle_count += 1

            if self.hold_angle_count <= self.max_hold_angle_frames:
                self.angle = self.prev_angle
            else:
                self.angle = self.prev_angle * self.hold_angle_decay
                self.prev_angle = self.angle

            self.speed = self.no_line_speed

        else:
            (
                lane_center,
                lane_center_near,
                target_x,
                center_error,
                heading_error,
                left_x,
                right_x,
                left_available,
                right_available,
                using_prediction,
                left_count,
                right_count
            ) = result

            self.hold_angle_count = 0
            self.holding_angle = False

            self.lane_available = True
            self.left_available = left_available
            self.right_available = right_available
            self.using_prediction = using_prediction

            self.last_left_x = left_x
            self.last_right_x = right_x
            self.last_center_x = lane_center
            self.last_center_near_x = lane_center_near
            self.last_target_x = target_x

            self.last_center_error = center_error
            self.last_heading_error = heading_error

            self.last_left_count = left_count
            self.last_right_count = right_count

            control_error = center_error + self.heading_gain * heading_error
            self.last_control_error = control_error

            dynamic_gain, dynamic_smoothing = self.get_dynamic_response(
                control_error
            )

            self.last_dynamic_gain = dynamic_gain
            self.last_dynamic_smoothing = dynamic_smoothing

            raw_angle = (
                self.steering_sign
                * dynamic_gain
                * control_error
            )

            raw_angle = self.clamp(
                raw_angle,
                -self.max_angle,
                self.max_angle
            )

            smoothed_angle = (
                dynamic_smoothing * self.prev_angle
                + (1.0 - dynamic_smoothing) * raw_angle
            )

            self.angle = self.clamp(
                smoothed_angle,
                -self.max_angle,
                self.max_angle
            )

            if using_prediction:
                self.speed = self.prediction_speed
            else:
                self.speed = self.line_detected_speed

            self.prev_angle = self.angle

        if self.show_debug:
            self.show_debug_view(image)

        self.print_log()

        return self.angle, self.speed

    # =====================================================
    # Dynamic response
    # =====================================================

    def get_dynamic_response(self, control_error):
        abs_error = abs(control_error)

        if abs_error < self.dead_zone_px:
            return 0.0, self.near_smoothing_alpha

        error_ratio = (
            (abs_error - self.near_error_px)
            / (self.far_error_px - self.near_error_px)
        )

        error_ratio = self.clamp(error_ratio, 0.0, 1.0)

        dynamic_gain = (
            self.min_center_gain
            + error_ratio * (self.max_center_gain - self.min_center_gain)
        )

        dynamic_smoothing = (
            self.near_smoothing_alpha
            - error_ratio * (
                self.near_smoothing_alpha - self.far_smoothing_alpha
            )
        )

        return dynamic_gain, dynamic_smoothing

    # =====================================================
    # Image warp
    # =====================================================

    def warp_image(self, img):
        h, w = img.shape[:2]

        src = np.float32([
            [int(w * 0.02), h],
            [int(w * 0.35), int(h * 0.55)],
            [int(w * 0.65), int(h * 0.55)],
            [int(w * 0.98), h]
        ])

        dst = np.float32([
            [int(w * 0.15), h],
            [int(w * 0.15), 0],
            [int(w * 0.85), 0],
            [int(w * 0.85), h]
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR
        )

        return warped

    # =====================================================
    # White lane center detection
    # =====================================================

    def detect_white_lane_center(self, image):
        warped = self.warp_image(image)

        height, width = warped.shape[:2]

        roi_y1 = int(height * self.roi_y_start_ratio)
        roi = warped[roi_y1:height, 0:width].copy()

        if roi.size == 0:
            return None

        white_mask = self.create_white_mask(roi)

        open_kernel = np.ones((3, 3), np.uint8)
        close_kernel = np.ones((7, 7), np.uint8)

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            open_kernel
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        self.last_warped = warped
        self.last_mask = white_mask
        self.last_roi_y1 = roi_y1

        ys, xs = white_mask.nonzero()

        if len(xs) < self.min_total_white_pixels:
            # 픽셀이 너무 적으면 현재 프레임 검출 실패
            return self.try_prediction_only(height, width)

        full_ys = ys + roi_y1
        full_xs = xs

        split_x = int(width * 0.50)

        left_filter = full_xs < split_x
        right_filter = full_xs >= split_x

        left_xs = full_xs[left_filter]
        left_ys = full_ys[left_filter]

        right_xs = full_xs[right_filter]
        right_ys = full_ys[right_filter]

        left_fit, left_current_detected = self.get_lane_fit(
            left_xs,
            left_ys,
            side="left"
        )

        right_fit, right_current_detected = self.get_lane_fit(
            right_xs,
            right_ys,
            side="right"
        )

        left_count = len(left_xs)
        right_count = len(right_xs)

        return self.compute_lane_center_from_fits(
            left_fit,
            right_fit,
            left_current_detected,
            right_current_detected,
            left_count,
            right_count,
            height,
            width
        )

    def try_prediction_only(self, height, width):
        left_fit = None
        right_fit = None

        left_predicted = False
        right_predicted = False

        if (
            self.last_left_fit is not None
            and self.left_prediction_count < self.max_side_prediction_frames
        ):
            left_fit = self.last_left_fit
            self.left_prediction_count += 1
            left_predicted = True

        if (
            self.last_right_fit is not None
            and self.right_prediction_count < self.max_side_prediction_frames
        ):
            right_fit = self.last_right_fit
            self.right_prediction_count += 1
            right_predicted = True

        if left_fit is None and right_fit is None:
            return None

        return self.compute_lane_center_from_fits(
            left_fit,
            right_fit,
            left_current_detected=False,
            right_current_detected=False,
            left_count=0,
            right_count=0,
            height=height,
            width=width,
            forced_prediction=(left_predicted or right_predicted)
        )

    def get_lane_fit(self, xs, ys, side):
        current_detected = len(xs) >= self.min_side_pixels

        if side == "left":
            last_fit = self.last_left_fit
        else:
            last_fit = self.last_right_fit

        if current_detected:
            try:
                new_fit = np.polyfit(ys, xs, 2)

                if last_fit is None:
                    fit = new_fit
                else:
                    fit = (
                        self.fit_alpha * last_fit
                        + (1.0 - self.fit_alpha) * new_fit
                    )

                if side == "left":
                    self.last_left_fit = fit
                    self.left_prediction_count = 0
                else:
                    self.last_right_fit = fit
                    self.right_prediction_count = 0

                return fit, True

            except Exception:
                pass

        # 현재 프레임에서 못 찾으면 이전 fit으로 잠깐 예측
        if side == "left":
            if (
                self.last_left_fit is not None
                and self.left_prediction_count < self.max_side_prediction_frames
            ):
                self.left_prediction_count += 1
                return self.last_left_fit, False

        else:
            if (
                self.last_right_fit is not None
                and self.right_prediction_count < self.max_side_prediction_frames
            ):
                self.right_prediction_count += 1
                return self.last_right_fit, False

        return None, False

    def compute_lane_center_from_fits(
        self,
        left_fit,
        right_fit,
        left_current_detected,
        right_current_detected,
        left_count,
        right_count,
        height,
        width,
        forced_prediction=False
    ):
        lookahead_y = int(height * self.lookahead_y_ratio)
        near_y = int(height * self.near_y_ratio)

        left_x = None
        right_x = None
        left_x_near = None
        right_x_near = None

        if left_fit is not None:
            left_x = self.eval_poly(left_fit, lookahead_y)
            left_x_near = self.eval_poly(left_fit, near_y)

            if not self.is_reasonable_x(left_x, width):
                left_x = None
                left_x_near = None

        if right_fit is not None:
            right_x = self.eval_poly(right_fit, lookahead_y)
            right_x_near = self.eval_poly(right_fit, near_y)

            if not self.is_reasonable_x(right_x, width):
                right_x = None
                right_x_near = None

        left_available = left_x is not None
        right_available = right_x is not None

        if not left_available and not right_available:
            return None

        # 양쪽 차선이 모두 보이면 차선 폭 업데이트
        if left_available and right_available:
            measured_width = right_x - left_x

            min_width = width * self.min_lane_width_ratio
            max_width = width * self.max_lane_width_ratio

            if min_width <= measured_width <= max_width:
                if self.estimated_lane_width_px is None:
                    self.estimated_lane_width_px = measured_width
                else:
                    self.estimated_lane_width_px = (
                        self.lane_width_alpha * self.estimated_lane_width_px
                        + (1.0 - self.lane_width_alpha) * measured_width
                    )

                self.last_lane_width = self.estimated_lane_width_px

        lane_width = self.get_lane_width_estimate(width)

        # 한쪽만 보이면 이전 차선 폭으로 반대쪽 예측
        if left_available and not right_available:
            right_x = left_x + lane_width
            right_x_near = left_x_near + lane_width
            right_available = True

        elif right_available and not left_available:
            left_x = right_x - lane_width
            left_x_near = right_x_near - lane_width
            left_available = True

        lane_center = (left_x + right_x) / 2.0
        lane_center_near = (left_x_near + right_x_near) / 2.0

        target_x = width * self.target_center_x_ratio

        center_error = target_x - lane_center
        heading_error = lane_center_near - lane_center

        using_prediction = (
            forced_prediction
            or not left_current_detected
            or not right_current_detected
        )

        return (
            float(lane_center),
            float(lane_center_near),
            float(target_x),
            float(center_error),
            float(heading_error),
            float(left_x),
            float(right_x),
            left_available,
            right_available,
            using_prediction,
            int(left_count),
            int(right_count)
        )

    def create_white_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

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

        white_mask = cv2.bitwise_and(
            hsv_white_mask,
            bgr_white_mask
        )

        return white_mask

    # =====================================================
    # Utility for lane model
    # =====================================================

    def eval_poly(self, fit, y):
        return float(fit[0] * y * y + fit[1] * y + fit[2])

    def is_reasonable_x(self, x, width):
        return -0.25 * width <= x <= 1.25 * width

    def get_lane_width_estimate(self, width):
        if self.estimated_lane_width_px is not None:
            return self.estimated_lane_width_px

        return width * self.default_lane_width_ratio

    # =====================================================
    # Debug view
    # =====================================================

    def show_debug_view(self, image):
        if self.last_warped is None:
            return

        original = image.copy()
        warped_debug = self.last_warped.copy()

        height, width = warped_debug.shape[:2]

        roi_y1 = int(height * self.roi_y_start_ratio)
        lookahead_y = int(height * self.lookahead_y_ratio)
        near_y = int(height * self.near_y_ratio)

        # ROI
        cv2.rectangle(
            warped_debug,
            (0, roi_y1),
            (width - 1, height - 1),
            (255, 255, 0),
            2
        )

        # 화면 중심
        target_x = int(width * self.target_center_x_ratio)
        cv2.line(
            warped_debug,
            (target_x, roi_y1),
            (target_x, height),
            (255, 255, 255),
            2
        )

        # lookahead / near
        cv2.line(
            warped_debug,
            (0, lookahead_y),
            (width, lookahead_y),
            (255, 255, 255),
            2
        )

        cv2.line(
            warped_debug,
            (0, near_y),
            (width, near_y),
            (120, 120, 120),
            2
        )

        # 왼쪽/오른쪽 fit 곡선
        self.draw_fit_curve(
            warped_debug,
            self.last_left_fit,
            roi_y1,
            height,
            color=(255, 0, 0)
        )

        self.draw_fit_curve(
            warped_debug,
            self.last_right_fit,
            roi_y1,
            height,
            color=(0, 255, 0)
        )

        # left/right/center point
        if self.last_left_x is not None:
            lx = int(self.clamp(self.last_left_x, 0, width - 1))
            cv2.circle(
                warped_debug,
                (lx, lookahead_y),
                8,
                (255, 0, 0),
                -1
            )

        if self.last_right_x is not None:
            rx = int(self.clamp(self.last_right_x, 0, width - 1))
            cv2.circle(
                warped_debug,
                (rx, lookahead_y),
                8,
                (0, 255, 0),
                -1
            )

        if self.last_center_x is not None:
            cx = int(self.clamp(self.last_center_x, 0, width - 1))

            cv2.line(
                warped_debug,
                (cx, roi_y1),
                (cx, height),
                (0, 0, 255),
                2
            )

            cv2.circle(
                warped_debug,
                (cx, lookahead_y),
                10,
                (0, 0, 255),
                -1
            )

        if self.holding_angle:
            status = "NO WHITE LANES - HOLD/DECAY"
            status_color = (0, 0, 255)
        elif self.using_prediction:
            status = "WHITE LANE PREDICTION"
            status_color = (0, 165, 255)
        elif self.lane_available:
            status = "WHITE LANE CENTER OK"
            status_color = (0, 255, 0)
        else:
            status = "NO WHITE LANES"
            status_color = (0, 0, 255)

        cv2.putText(
            warped_debug,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            status_color,
            2
        )

        cv2.putText(
            warped_debug,
            f"ANGLE:{self.angle:.2f} SPEED:{self.speed:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )

        if self.last_center_error is not None:
            cv2.putText(
                warped_debug,
                f"L:{self.left_available} R:{self.right_available} "
                f"CENTER:{self.last_center_x:.1f} "
                f"TARGET:{self.last_target_x:.1f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                warped_debug,
                f"ERR:{self.last_center_error:.1f} "
                f"HEAD:{self.last_heading_error:.1f} "
                f"CTRL:{self.last_control_error:.1f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                warped_debug,
                f"GAIN:{self.last_dynamic_gain:.3f} "
                f"SMOOTH:{self.last_dynamic_smoothing:.2f} "
                f"L_CNT:{self.last_left_count} R_CNT:{self.last_right_count}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2
            )

        else:
            cv2.putText(
                warped_debug,
                f"HOLD:{self.hold_angle_count} "
                f"PREV_ANGLE:{self.prev_angle:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

        # 마스크를 전체 크기로 복원
        mask_view = np.zeros((height, width), dtype=np.uint8)

        if self.last_mask is not None and self.last_roi_y1 is not None:
            mask_h, mask_w = self.last_mask.shape[:2]
            mask_view[
                self.last_roi_y1:self.last_roi_y1 + mask_h,
                0:mask_w
            ] = self.last_mask

        mask_color = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)

        view_w = 320
        view_h = 240

        original_small = cv2.resize(original, (view_w, view_h))
        warped_small = cv2.resize(warped_debug, (view_w, view_h))
        mask_small = cv2.resize(mask_color, (view_w, view_h))

        top = np.hstack((original_small, warped_small))
        bottom = np.hstack((mask_small, mask_small))
        combined = np.vstack((top, bottom))

        cv2.imshow(self.window_name, combined)
        cv2.waitKey(1)

    def draw_fit_curve(self, image, fit, y_start, y_end, color):
        if fit is None:
            return

        height, width = image.shape[:2]

        points = []

        for y in range(y_start, y_end, 10):
            x = self.eval_poly(fit, y)

            if 0 <= x < width:
                points.append((int(x), int(y)))

        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

            cv2.polylines(
                image,
                [pts],
                isClosed=False,
                color=color,
                thickness=4
            )

    # =====================================================
    # Basic utility
    # =====================================================

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def print_log(self):
        self.log_count += 1

        if self.log_count % 20 != 0:
            return

        center_text = (
            "None"
            if self.last_center_x is None
            else f"{self.last_center_x:.1f}"
        )

        err_text = (
            "None"
            if self.last_center_error is None
            else f"{self.last_center_error:.1f}"
        )

        ctrl_text = (
            "None"
            if self.last_control_error is None
            else f"{self.last_control_error:.1f}"
        )

        lane_width_text = (
            "None"
            if self.last_lane_width is None
            else f"{self.last_lane_width:.1f}"
        )

        self.log_info(
            f"AUTO_DRIVE WHITE | "
            f"lane:{self.lane_available} | "
            f"L:{self.left_available} | "
            f"R:{self.right_available} | "
            f"pred:{self.using_prediction} | "
            f"hold:{self.holding_angle} | "
            f"center:{center_text} | "
            f"err:{err_text} | "
            f"ctrl:{ctrl_text} | "
            f"width:{lane_width_text} | "
            f"gain:{self.last_dynamic_gain:.3f} | "
            f"smooth:{self.last_dynamic_smoothing:.2f} | "
            f"angle:{self.angle:.2f} | "
            f"speed:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)
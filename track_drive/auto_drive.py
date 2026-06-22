#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class AutoDrive:
    """
    흰색 차선 중앙 추종 AUTO_DRIVE.

    핵심 변경:
        - 작은 오차에서는 핸들을 새로 꺾지 않고 기존 각도를 서서히 줄임
        - 큰 오차가 생기는 순간 강하게 조향
        - hysteresis 적용:
            turn_start_error_px 이상이면 조향 시작
            turn_release_error_px 이하로 줄어들면 조향 해제
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE Lane Only"

        # =====================================================
        # Drive parameters
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        self.drive_speed = 4.0
        self.max_angle = 100.0
        self.prev_angle = 0.0

        # =====================================================
        # ROI / control parameters
        # =====================================================
        self.roi_y_start_ratio = 0.30
        self.lookahead_y_ratio = 0.60
        self.near_y_ratio = 0.95

        self.target_center_x_ratio = 0.50

        # 차선 진행 방향 보정
        self.heading_gain = 0.40

        # 조향 방향 보정
        # 차가 반대로 꺾이면 1.0으로 변경
        self.steering_sign = -1.0

        # =====================================================
        # Threshold steering mode
        # =====================================================
        # 오차가 이 값보다 작으면 새 조향을 하지 않음
        self.turn_start_error_px = 45.0

        # 조향 중이었다가 오차가 이 값보다 작아지면 조향 해제
        self.turn_release_error_px = 18.0

        # 작은 오차 구간에서 이전 핸들 각도 감쇠
        # 1.0이면 완전 유지, 0.90이면 서서히 직진 복귀
        self.small_error_angle_decay = 0.90

        # 큰 오차에서 조향 강도
        self.threshold_turn_gain = 0.95

        # 큰 오차에서 최소 조향각
        self.min_turn_angle = 18.0

        # 큰 오차에서 smoothing
        # 작을수록 즉시 꺾음
        self.turn_smoothing_alpha = 0.08

        # 현재 강제 조향 상태
        self.correction_active = False

        # =====================================================
        # White threshold
        # =====================================================
        self.white_lower = np.array([0, 0, 155])
        self.white_upper = np.array([180, 95, 255])
        self.min_bgr_white = 135

        self.min_total_white_pixels = 120
        self.min_side_pixels = 55

        # =====================================================
        # Sliding window parameters
        # =====================================================
        self.nwindows = 7
        self.window_margin = 70
        self.minpix_recenter = 20
        self.hist_peak_min = 8

        # =====================================================
        # Lane prediction parameters
        # =====================================================
        self.last_left_fit = None
        self.last_right_fit = None

        self.fit_alpha = 0.40

        self.left_prediction_count = 0
        self.right_prediction_count = 0
        self.max_side_prediction_frames = 8

        self.default_lane_width_ratio = 0.52
        self.estimated_lane_width_px = None
        self.lane_width_alpha = 0.75

        self.min_lane_width_ratio = 0.25
        self.max_lane_width_ratio = 0.90

        self.last_good_center_x = None
        self.max_center_jump_px = 90.0

        # 둘 다 안 보일 때
        self.hold_angle_count = 0
        self.max_hold_angle_frames = 6
        self.hold_angle_decay = 0.70

        # =====================================================
        # State / debug data
        # =====================================================
        self.started = False
        self.warned_no_image = False

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
        self.last_windows = []

    # =====================================================
    # Lifecycle
    # =====================================================

    def start(self):
        if self.started:
            return

        self.started = True

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

    # =====================================================
    # Main process
    # =====================================================

    def process(self, image):
        if image is None:
            self.angle = self.prev_angle
            self.speed = self.drive_speed
            return self.angle, self.speed

        result = self.detect_white_lane_center(image)

        if result is None:
            self.lane_available = False
            self.left_available = False
            self.right_available = False
            self.using_prediction = False
            self.holding_angle = True
            self.correction_active = False

            self.last_left_x = None
            self.last_right_x = None
            self.last_center_x = None
            self.last_center_near_x = None
            self.last_target_x = None

            self.last_center_error = None
            self.last_heading_error = None
            self.last_control_error = None

            self.hold_angle_count += 1

            if self.hold_angle_count <= self.max_hold_angle_frames:
                self.angle = self.prev_angle
            else:
                self.angle = self.prev_angle * self.hold_angle_decay
                self.prev_angle = self.angle

            self.speed = self.drive_speed

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

            # =================================================
            # 작은 오차는 유지, 큰 오차만 강하게 조향
            # =================================================
            self.angle = self.compute_threshold_steering(control_error)

            self.prev_angle = self.angle
            self.speed = self.drive_speed

        if self.show_debug:
            self.show_debug_view()

        return float(self.angle), float(self.speed)

    # =====================================================
    # Threshold steering
    # =====================================================

    def compute_threshold_steering(self, control_error):
        abs_error = abs(control_error)

        # =====================================================
        # 1. 조향 상태 ON/OFF 판단
        # =====================================================
        if self.correction_active:
            # 이미 조향 중이면 오차가 충분히 작아질 때까지 유지
            if abs_error <= self.turn_release_error_px:
                self.correction_active = False
        else:
            # 오차가 충분히 커졌을 때만 조향 시작
            if abs_error >= self.turn_start_error_px:
                self.correction_active = True

        # =====================================================
        # 2. 오차가 작으면 새 조향을 하지 않음
        # =====================================================
        if not self.correction_active:
            keep_angle = self.prev_angle * self.small_error_angle_decay

            return self.clamp(
                keep_angle,
                -self.max_angle,
                self.max_angle
            )

        # =====================================================
        # 3. 오차가 크면 강하게 조향
        # =====================================================
        raw_angle = (
            self.steering_sign
            * self.threshold_turn_gain
            * control_error
        )

        # 큰 오차인데 조향각이 너무 작게 나오는 것 방지
        if abs(raw_angle) < self.min_turn_angle:
            if raw_angle > 0:
                raw_angle = self.min_turn_angle
            elif raw_angle < 0:
                raw_angle = -self.min_turn_angle

        raw_angle = self.clamp(
            raw_angle,
            -self.max_angle,
            self.max_angle
        )

        # 큰 오차에서는 빠르게 반영
        angle = (
            self.turn_smoothing_alpha * self.prev_angle
            + (1.0 - self.turn_smoothing_alpha) * raw_angle
        )

        return self.clamp(
            angle,
            -self.max_angle,
            self.max_angle
        )

    # =====================================================
    # Bird's Eye View
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
    # Lane detection
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
        self.last_windows = []

        ys_roi, xs = white_mask.nonzero()

        if len(xs) < self.min_total_white_pixels:
            return self.try_prediction_only(height, width)

        full_ys = ys_roi + roi_y1

        left_base, right_base = self.find_lane_bases(white_mask, width)

        left_fit, left_current_detected, left_count = self.fit_lane_with_sliding_window(
            xs,
            ys_roi,
            full_ys,
            base_x=left_base,
            side="left",
            roi_y1=roi_y1,
            height=height,
            width=width
        )

        right_fit, right_current_detected, right_count = self.fit_lane_with_sliding_window(
            xs,
            ys_roi,
            full_ys,
            base_x=right_base,
            side="right",
            roi_y1=roi_y1,
            height=height,
            width=width
        )

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

    def find_lane_bases(self, white_mask, width):
        roi_h = white_mask.shape[0]

        lower_half = white_mask[roi_h // 2:, :]
        histogram = np.sum(lower_half > 0, axis=0)

        midpoint = width // 2

        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]

        left_base = None
        right_base = None

        if len(left_hist) > 0 and np.max(left_hist) >= self.hist_peak_min:
            left_base = int(np.argmax(left_hist))

        if len(right_hist) > 0 and np.max(right_hist) >= self.hist_peak_min:
            right_base = int(np.argmax(right_hist) + midpoint)

        return left_base, right_base

    def fit_lane_with_sliding_window(
        self,
        nonzerox,
        nonzeroy_roi,
        full_nonzeroy,
        base_x,
        side,
        roi_y1,
        height,
        width
    ):
        if base_x is None:
            return self.try_side_prediction(side)

        roi_h = height - roi_y1
        window_height = int(roi_h / self.nwindows)

        current_x = base_x
        lane_inds = []

        for window in range(self.nwindows):
            win_y_low_roi = roi_h - (window + 1) * window_height
            win_y_high_roi = roi_h - window * window_height

            win_x_low = current_x - self.window_margin
            win_x_high = current_x + self.window_margin

            self.last_windows.append(
                (
                    win_x_low,
                    win_y_low_roi + roi_y1,
                    win_x_high,
                    win_y_high_roi + roi_y1,
                    side
                )
            )

            good_inds = (
                (nonzeroy_roi >= win_y_low_roi)
                & (nonzeroy_roi < win_y_high_roi)
                & (nonzerox >= win_x_low)
                & (nonzerox < win_x_high)
            ).nonzero()[0]

            lane_inds.append(good_inds)

            if len(good_inds) > self.minpix_recenter:
                current_x = int(np.mean(nonzerox[good_inds]))

        if len(lane_inds) == 0:
            return self.try_side_prediction(side)

        lane_inds = np.concatenate(lane_inds)

        pixel_count = len(lane_inds)

        if pixel_count < self.min_side_pixels:
            return self.try_side_prediction(side)

        lane_xs = nonzerox[lane_inds]
        lane_ys = full_nonzeroy[lane_inds]

        try:
            new_fit = np.polyfit(lane_ys, lane_xs, 2)

            if side == "left":
                if self.last_left_fit is None:
                    fit = new_fit
                else:
                    fit = (
                        self.fit_alpha * self.last_left_fit
                        + (1.0 - self.fit_alpha) * new_fit
                    )

                self.last_left_fit = fit
                self.left_prediction_count = 0

            else:
                if self.last_right_fit is None:
                    fit = new_fit
                else:
                    fit = (
                        self.fit_alpha * self.last_right_fit
                        + (1.0 - self.fit_alpha) * new_fit
                    )

                self.last_right_fit = fit
                self.right_prediction_count = 0

            return fit, True, pixel_count

        except Exception:
            return self.try_side_prediction(side)

    def try_side_prediction(self, side):
        if side == "left":
            if (
                self.last_left_fit is not None
                and self.left_prediction_count < self.max_side_prediction_frames
            ):
                self.left_prediction_count += 1
                return self.last_left_fit, False, 0

        else:
            if (
                self.last_right_fit is not None
                and self.right_prediction_count < self.max_side_prediction_frames
            ):
                self.right_prediction_count += 1
                return self.last_right_fit, False, 0

        return None, False, 0

    def try_prediction_only(self, height, width):
        left_fit, left_current_detected, left_count = self.try_side_prediction("left")
        right_fit, right_current_detected, right_count = self.try_side_prediction("right")

        if left_fit is None and right_fit is None:
            return None

        return self.compute_lane_center_from_fits(
            left_fit,
            right_fit,
            left_current_detected,
            right_current_detected,
            left_count,
            right_count,
            height,
            width,
            forced_prediction=True
        )

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

        lane_width = None

        if left_available and right_available:
            lane_width = right_x - left_x

            min_width = width * self.min_lane_width_ratio
            max_width = width * self.max_lane_width_ratio

            if not (min_width <= lane_width <= max_width):
                if left_count >= right_count and left_available:
                    right_available = False
                    right_x = None
                    right_x_near = None
                elif right_available:
                    left_available = False
                    left_x = None
                    left_x_near = None

            else:
                if self.estimated_lane_width_px is None:
                    self.estimated_lane_width_px = lane_width
                else:
                    self.estimated_lane_width_px = (
                        self.lane_width_alpha * self.estimated_lane_width_px
                        + (1.0 - self.lane_width_alpha) * lane_width
                    )

                self.last_lane_width = self.estimated_lane_width_px

        estimated_width = self.get_lane_width_estimate(width)

        if left_available and not right_available:
            right_x = left_x + estimated_width
            right_x_near = left_x_near + estimated_width
            right_available = True

        elif right_available and not left_available:
            left_x = right_x - estimated_width
            left_x_near = right_x_near - estimated_width
            left_available = True

        if left_x is None or right_x is None:
            return None

        lane_center = (left_x + right_x) / 2.0
        lane_center_near = (left_x_near + right_x_near) / 2.0

        if self.last_good_center_x is not None:
            center_diff = lane_center - self.last_good_center_x

            if abs(center_diff) > self.max_center_jump_px:
                lane_center = (
                    self.last_good_center_x
                    + np.sign(center_diff) * self.max_center_jump_px
                )

        self.last_good_center_x = lane_center

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

    # =====================================================
    # Utility
    # =====================================================

    def eval_poly(self, fit, y):
        return float(fit[0] * y * y + fit[1] * y + fit[2])

    def is_reasonable_x(self, x, width):
        return -0.25 * width <= x <= 1.25 * width

    def get_lane_width_estimate(self, width):
        if self.estimated_lane_width_px is not None:
            return self.estimated_lane_width_px

        return width * self.default_lane_width_ratio

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    # =====================================================
    # Debug view: lane only
    # =====================================================

    def show_debug_view(self):
        if self.last_warped is None:
            return

        height, width = self.last_warped.shape[:2]

        lane_view = np.zeros((height, width, 3), dtype=np.uint8)

        if self.last_mask is not None and self.last_roi_y1 is not None:
            mask_h, mask_w = self.last_mask.shape[:2]

            mask_color = cv2.cvtColor(self.last_mask, cv2.COLOR_GRAY2BGR)

            lane_view[
                self.last_roi_y1:self.last_roi_y1 + mask_h,
                0:mask_w
            ] = mask_color

        roi_y1 = int(height * self.roi_y_start_ratio)
        lookahead_y = int(height * self.lookahead_y_ratio)

        for win_x_low, win_y_low, win_x_high, win_y_high, side in self.last_windows:
            color = (255, 0, 0) if side == "left" else (0, 255, 0)

            cv2.rectangle(
                lane_view,
                (int(win_x_low), int(win_y_low)),
                (int(win_x_high), int(win_y_high)),
                color,
                1
            )

        self.draw_fit_curve(
            lane_view,
            self.last_left_fit,
            roi_y1,
            height,
            color=(255, 0, 0)
        )

        self.draw_fit_curve(
            lane_view,
            self.last_right_fit,
            roi_y1,
            height,
            color=(0, 255, 0)
        )

        target_x = int(width * self.target_center_x_ratio)

        cv2.line(
            lane_view,
            (target_x, roi_y1),
            (target_x, height),
            (255, 255, 255),
            2
        )

        if self.last_center_x is not None:
            cx = int(self.clamp(self.last_center_x, 0, width - 1))

            cv2.line(
                lane_view,
                (cx, roi_y1),
                (cx, height),
                (0, 0, 255),
                2
            )

            cv2.circle(
                lane_view,
                (cx, lookahead_y),
                8,
                (0, 0, 255),
                -1
            )

        if self.last_left_x is not None:
            lx = int(self.clamp(self.last_left_x, 0, width - 1))
            cv2.circle(
                lane_view,
                (lx, lookahead_y),
                6,
                (255, 0, 0),
                -1
            )

        if self.last_right_x is not None:
            rx = int(self.clamp(self.last_right_x, 0, width - 1))
            cv2.circle(
                lane_view,
                (rx, lookahead_y),
                6,
                (0, 255, 0),
                -1
            )

        cv2.imshow(self.window_name, lane_view)
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
                thickness=3
            )

    def log_info(self, msg):
        pass

    def log_warn(self, msg):
        pass
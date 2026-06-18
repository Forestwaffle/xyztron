#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rclpy
import math
import numpy as np
import matplotlib.pyplot as plt

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

from xycar_msgs.msg import XycarMotor
from track_drive.traffic_light_detector import TrafficLightDetector


class MainDrivingNode(Node):
    def __init__(self):
        super().__init__('main_driving_node')

        self.get_logger().info("Main driving node started")

        self.bridge = CvBridge()
        self.image = None

        self.motor_msg = XycarMotor()

        self.qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # =====================================================
        # Camera starts first.
        # LiDAR is NOT subscribed here.
        # =====================================================
        self.camera_sub = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.camera_callback,
            self.qos
        )

        # LiDAR objects.
        # They will be initialized only after CONE_DRIVE starts.
        self.lidar_sub = None
        self.lidar_started = False
        self.lidar_ranges = None
        self.lidar_data_received = False

        # LiDAR debug viewer objects
        self.lidar_viewer_ready = False
        self.lidar_fig = None
        self.lidar_ax = None
        self.lidar_points = None
        self.lidar_log_counter = 0
        self.warned_no_lidar = False

        # Motor publisher
        self.motor_pub = self.create_publisher(
            XycarMotor,
            '/xycar_motor',
            10
        )

        # Traffic light detector
        self.traffic_light = TrafficLightDetector(show_debug=True)

        # Mission state
        self.mission_state = "WAIT_TRAFFIC_LIGHT"

        # Previous states for terminal logging
        self.prev_traffic_light_state = None
        self.prev_mission_state = None

        # Main loop: 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)

    def camera_callback(self, msg):
        """Store latest front camera image."""
        try:
            self.image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f"Camera conversion failed: {e}")

    def start_lidar(self):
        """Start LiDAR subscriber after mission state changes to CONE_DRIVE."""
        if self.lidar_started:
            return

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            self.qos
        )

        self.lidar_started = True
        self.get_logger().info("LiDAR subscriber started after CONE_DRIVE")

    def lidar_callback(self, msg):
        """Store latest LiDAR scan data."""
        self.lidar_ranges = msg.ranges

        if not self.lidar_data_received:
            self.get_logger().info(
                f"LiDAR data received: {len(msg.ranges)} beams"
            )
            self.lidar_data_received = True

    def init_lidar_viewer(self):
        """Create LiDAR debug viewer window once."""
        if self.lidar_viewer_ready:
            return

        self.lidar_fig, self.lidar_ax = plt.subplots(figsize=(8, 8))
        self.lidar_ax.set_title("LiDAR Debug Viewer")
        self.lidar_ax.set_aspect('equal')
        self.lidar_ax.set_xlim(-10, 10)
        self.lidar_ax.set_ylim(-10, 10)
        self.lidar_ax.grid(True)

        # LiDAR points
        self.lidar_points = self.lidar_ax.scatter([], [], s=5)

        # Vehicle center
        self.lidar_ax.plot(0, 0, 'ro')

        # Front direction marker
        self.lidar_ax.plot([0, 0], [0, 2], 'r-')
        self.lidar_ax.text(0.2, 2.0, "FRONT")

        plt.ion()
        plt.show(block=False)

        self.lidar_viewer_ready = True
        self.get_logger().info("LiDAR debug viewer started")

    def update_lidar_viewer(self):
        """Update LiDAR debug viewer with latest scan."""
        if self.lidar_ranges is None:
            if not self.warned_no_lidar:
                self.get_logger().warn("No LiDAR data yet")
                self.warned_no_lidar = True
            return

        ranges = np.array([
            d if math.isfinite(d) else np.nan
            for d in self.lidar_ranges
        ], dtype=float)

        if len(ranges) == 0:
            return

        # Coordinate assumption:
        # index 0 = left, index 90 = front
        angles = np.deg2rad(np.arange(len(ranges)) - 90)

        x = -ranges * np.cos(angles)
        y = -ranges * np.sin(angles)

        valid_mask = np.isfinite(x) & np.isfinite(y)

        valid_x = x[valid_mask]
        valid_y = y[valid_mask]

        indices = np.arange(len(ranges))
        colors = np.full(len(ranges), 'b', dtype=object)

        # Debug color sectors
        colors[(indices >= 0) & (indices < 45)] = 'r'
        colors[(indices >= 45) & (indices < 90)] = 'g'
        colors[(indices >= 90) & (indices < 270)] = 'b'
        colors[(indices >= 270) & (indices < 315)] = 'orange'
        colors[(indices >= 315) & (indices < 360)] = 'purple'

        valid_colors = colors[valid_mask]

        self.lidar_points.set_offsets(np.c_[valid_x, valid_y])
        self.lidar_points.set_color(valid_colors)

        self.lidar_fig.canvas.draw_idle()
        self.lidar_fig.canvas.flush_events()

        # Terminal debug log, about 1 Hz because control_loop is 20 Hz
        self.lidar_log_counter += 1

        if self.lidar_log_counter % 20 == 0:
            front_candidates = []

            if len(self.lidar_ranges) >= 95:
                front_candidates = [
                    d for d in self.lidar_ranges[85:95]
                    if math.isfinite(d)
                ]

            if front_candidates:
                front = min(front_candidates)
                self.get_logger().info(
                    f"LiDAR front distance: {front:.2f} m"
                )

    def drive(self, angle, speed):
        """Publish motor command."""
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    def log_mission_state_changed(self):
        """Log mission state only when it changes."""
        if self.mission_state != self.prev_mission_state:
            self.get_logger().info(
                f"Mission state changed: {self.mission_state}"
            )
            self.prev_mission_state = self.mission_state

    def log_traffic_light_state_changed(self, state):
        """Log traffic light state only when it changes."""
        if state != self.prev_traffic_light_state:
            self.get_logger().info(
                f"Traffic light state changed: {state}"
            )
            self.prev_traffic_light_state = state

    def control_loop(self):
        """Main mission logic."""
        if self.image is None:
            self.drive(0, 0)
            return

        self.log_mission_state_changed()

        # =====================================================
        # WAIT_TRAFFIC_LIGHT:
        #   - Camera is used first
        #   - Show only Traffic Light Detector window
        #   - Stop on RED
        #   - Switch to GO on GREEN
        # =====================================================
        if self.mission_state == "WAIT_TRAFFIC_LIGHT":
            if not self.traffic_light.active:
                self.traffic_light.enable()

            state, detected_light, debug_frame = self.traffic_light.process(
                self.image
            )

            self.log_traffic_light_state_changed(state)

            if state == "STOP":
                self.drive(0, 0)

            elif state == "GO":
                # Stop once before changing mission
                self.drive(0, 0)

                # Close Traffic Light Detector window
                self.traffic_light.disable()

                # Move to next mission
                self.mission_state = "CONE_DRIVE"

                self.get_logger().info("Traffic light mission complete")

        # =====================================================
        # CONE_DRIVE:
        #   - LiDAR starts only after this state begins
        #   - Show LiDAR debug viewer
        #   - Vehicle stays stopped for now
        # =====================================================
        elif self.mission_state == "CONE_DRIVE":
            self.start_lidar()
            self.init_lidar_viewer()
            self.update_lidar_viewer()

            # No cone-driving logic yet. Keep stopped for safety.
            self.drive(0, 0)

        # =====================================================
        # Safety stop
        # =====================================================
        else:
            self.drive(0, 0)


def main(args=None):
    rclpy.init(args=args)

    node = MainDrivingNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.drive(0, 0)
        cv2.destroyAllWindows()
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
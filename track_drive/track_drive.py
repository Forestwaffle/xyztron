#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
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

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Front camera subscriber
        self.camera_sub = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.camera_callback,
            qos
        )

        # Motor publisher
        self.motor_pub = self.create_publisher(
            XycarMotor,
            '/xycar_motor',
            10
        )

        # Traffic light function class
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

    def drive(self, angle, speed):
        """Publish motor command."""
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    def show_camera(self, text=""):
        """Keep normal camera window updating after traffic light mission."""
        if self.image is None:
            return

        frame = self.image.copy()

        if text:
            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            f"MISSION: {self.mission_state}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.imshow("Main Camera", frame)
        cv2.waitKey(1)

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
        # Mission 1: wait for green traffic light
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
                self.drive(0, 0)

                self.traffic_light.disable()
                self.mission_state = "CONE_DRIVE"

                self.get_logger().info("Traffic light mission complete")

        # =====================================================
        # Mission 2: cone driving placeholder
        # =====================================================
        elif self.mission_state == "CONE_DRIVE":
            # Keep camera window updating.
            # Cone driving logic will be added here later.
            self.drive(0, 0)
            self.show_camera("CONE_DRIVE MODE")

        # =====================================================
        # Default safety stop
        # =====================================================
        else:
            self.drive(0, 0)
            self.show_camera("UNKNOWN MODE")


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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
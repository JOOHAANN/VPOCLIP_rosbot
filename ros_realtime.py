#!/usr/bin/env python3
"""Run CLIPGCN on the latest image received from a ROS 2 camera topic."""

import re
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from webcam_realtime import build_arg_parser, main as run_realtime


DEFAULT_ROBOT_NAMESPACE = "rosbot_1"


def normalize_robot_namespace(value):
    namespace = str(value).strip().strip("/")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", namespace):
        raise ValueError(
            "--robot-namespace must be one ROS name such as rosbot_1 or rosbot_2."
        )
    return namespace


def build_ros_arg_parser():
    parser = build_arg_parser("Run CLIPGCN realtime inference from a ROS 2 image topic.")
    parser.add_argument(
        "--robot-namespace",
        default=DEFAULT_ROBOT_NAMESPACE,
        help=(
            "Target ROSbot namespace. It supplies the default camera and "
            "PersonDetection topics (for example rosbot_1 or rosbot_2)."
        ),
    )
    parser.add_argument(
        "--image-topic",
        default=None,
        help="ROS 2 camera topic. Defaults below --robot-namespace.",
    )
    parser.add_argument(
        "--image-transport",
        choices=["compressed", "raw"],
        default="compressed",
        help="Message type carried by --image-topic.",
    )
    parser.add_argument(
        "--ros-qos-reliability",
        choices=["reliable", "best-effort"],
        default="reliable",
        help="Subscription reliability. It should match the camera publisher.",
    )
    parser.add_argument(
        "--ros-wait-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the next camera frame before reporting an error.",
    )
    parser.add_argument(
        "--ros-node-name",
        default="clipgcn_realtime",
        help="ROS 2 node name.",
    )
    parser.add_argument(
        "--enable-person-follow-output",
        action="store_true",
        help=(
            "Publish the selected YOLO person on "
            "<robot-namespace>/follow/person_detection for the PID controller."
        ),
    )
    parser.add_argument(
        "--person-detection-topic",
        default=None,
        help="PersonDetection output topic. Defaults below --robot-namespace.",
    )
    parser.add_argument(
        "--person-detection-frame-id",
        default=None,
        help="Header frame ID. Defaults to <robot-namespace>/camera_rgb_optical_frame.",
    )
    parser.add_argument(
        "--person-confidence-threshold",
        type=float,
        default=0.35,
        help="Minimum YOLO confidence for acquiring/tracking a person.",
    )
    parser.add_argument(
        "--person-lock-iou-threshold",
        type=float,
        default=0.15,
        help="Minimum IoU that keeps a person associated with the current lock.",
    )
    parser.add_argument(
        "--person-max-center-jump",
        type=float,
        default=0.20,
        help="Maximum target center jump as a fraction of the image diagonal.",
    )
    parser.add_argument(
        "--person-max-lost-frames",
        type=int,
        default=5,
        help="Missing frames tolerated before a different person may be acquired.",
    )
    return parser


def validate_ros_args(args):
    args.robot_namespace = normalize_robot_namespace(args.robot_namespace)
    if not args.image_topic.startswith("/"):
        raise ValueError("--image-topic must be an absolute ROS topic name.")
    if args.ros_wait_timeout <= 0:
        raise ValueError("--ros-wait-timeout must be positive.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.ros_node_name):
        raise ValueError("--ros-node-name must contain only letters, digits, and underscores.")
    if not args.person_detection_topic.startswith("/"):
        raise ValueError("--person-detection-topic must be an absolute ROS topic name.")
    if not 0.0 <= args.person_confidence_threshold <= 1.0:
        raise ValueError("--person-confidence-threshold must be in [0, 1].")
    if not 0.0 <= args.person_lock_iou_threshold <= 1.0:
        raise ValueError("--person-lock-iou-threshold must be in [0, 1].")
    if args.person_max_center_jump < 0.0:
        raise ValueError("--person-max-center-jump must be non-negative.")
    if args.person_max_lost_frames < 0:
        raise ValueError("--person-max-lost-frames must be non-negative.")


def qos_profile(reliability):
    policy = (
        ReliabilityPolicy.RELIABLE
        if reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=policy,
        durability=DurabilityPolicy.VOLATILE,
    )


def decode_compressed_image(message):
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError(f"OpenCV could not decode compressed image format {message.format!r}.")
    return np.ascontiguousarray(frame_bgr)


def decode_raw_image(message):
    encoding = message.encoding.lower()
    channel_counts = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
    }
    if encoding not in channel_counts:
        raise ValueError(
            f"Unsupported raw ROS image encoding {message.encoding!r}; "
            f"supported encodings are {sorted(channel_counts)}."
        )

    channels = channel_counts[encoding]
    packed_width = int(message.width) * channels
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(int(message.height), int(message.step))
    pixels = rows[:, :packed_width]
    if channels == 1:
        image = pixels.reshape(int(message.height), int(message.width))
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = pixels.reshape(int(message.height), int(message.width), channels)
    conversions = {
        "rgb8": cv2.COLOR_RGB2BGR,
        "bgra8": cv2.COLOR_BGRA2BGR,
        "rgba8": cv2.COLOR_RGBA2BGR,
    }
    if encoding in conversions:
        image = cv2.cvtColor(image, conversions[encoding])
    return np.ascontiguousarray(image)


class RosImageSource:
    """VideoCapture-like ROS source with a depth-one, latest-frame subscription."""

    source_name = "ROS 2 camera"

    def __init__(self, args):
        self.args = args
        self.source_description = (
            f"{args.image_topic} ({args.image_transport}, "
            f"QoS={args.ros_qos_reliability}, depth=1)"
        )
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)

        self.node = Node(args.ros_node_name)
        self._latest_frame = None
        self._latest_stamp = None
        self.latest_stamp = None
        self._received_sequence = 0
        self._delivered_sequence = 0
        self._decode_error = None
        message_type = CompressedImage if args.image_transport == "compressed" else Image
        self.subscription = self.node.create_subscription(
            message_type,
            args.image_topic,
            self._on_image,
            qos_profile(args.ros_qos_reliability),
        )

    def _on_image(self, message):
        try:
            frame = (
                decode_compressed_image(message)
                if self.args.image_transport == "compressed"
                else decode_raw_image(message)
            )
        except Exception as exc:
            self._decode_error = exc
            return

        self._latest_frame = frame
        self._latest_stamp = message.header.stamp
        self._received_sequence += 1
        self._decode_error = None

    def read(self):
        deadline = time.monotonic() + self.args.ros_wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            rclpy.spin_once(self.node, timeout_sec=min(0.25, remaining))
            if self._decode_error is not None:
                error = self._decode_error
                self._decode_error = None
                raise RuntimeError(f"Failed to decode {self.args.image_topic}: {error}") from error
            if (
                self._latest_frame is not None
                and self._received_sequence != self._delivered_sequence
            ):
                self._delivered_sequence = self._received_sequence
                self.latest_stamp = self._latest_stamp
                return True, self._latest_frame

        if not rclpy.ok():
            return False, None
        raise TimeoutError(
            f"No image received from {self.args.image_topic} within "
            f"{self.args.ros_wait_timeout:g} seconds. Check ROS_DOMAIN_ID, "
            "CycloneDDS network settings, topic name, and QoS."
        )

    def release(self):
        if getattr(self, "node", None) is not None:
            self.node.destroy_node()
            self.node = None
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()


def parse_args(argv=None):
    args = build_ros_arg_parser().parse_args(argv)
    args.robot_namespace = normalize_robot_namespace(args.robot_namespace)
    namespace_prefix = f"/{args.robot_namespace}"
    if args.image_topic is None:
        image_suffix = (
            "camera/rgb/image_raw/compressed"
            if args.image_transport == "compressed"
            else "camera/rgb/image_raw"
        )
        args.image_topic = f"{namespace_prefix}/{image_suffix}"
    if args.person_detection_topic is None:
        args.person_detection_topic = f"{namespace_prefix}/follow/person_detection"
    if args.person_detection_frame_id is None:
        args.person_detection_frame_id = (
            f"{args.robot_namespace}/camera_rgb_optical_frame"
        )
    validate_ros_args(args)
    return args


def main():
    run_realtime(args=parse_args(), capture_factory=RosImageSource)


if __name__ == "__main__":
    main()

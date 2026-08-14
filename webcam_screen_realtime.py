#!/usr/bin/env python
"""Realtime CLIPGCN inference from a screen region.

This entrypoint keeps the original webcam pipeline intact, but replaces
cv2.VideoCapture with Pillow ImageGrab. It is intended for running inside the
Docker container while a video is playing on the host desktop.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import ImageGrab

from action_label_utils import load_action_display_names
from test import load_split_classes, load_split_metadata
from test_raw_end_to_end import (
    CTRGCN_ROOT,
    X3D_ROOT,
    ObjectMapRunner,
    load_clipgcn_model,
    load_ctrgcn_model,
    load_x3d_model,
    load_yolo_model,
)
from train import get_device, get_path_from_config, load_config, print_device_info
from webcam_realtime import (
    DEFAULT_ACTION_DURATION_TYPES,
    MediaPipePoseSource,
    RTMPoseSource,
    build_decision_gate,
    active_window_seconds,
    default_mediapipe_model_asset,
    default_yolo_repo,
    default_yolo_weights,
    draw_overlay,
    finish_end_to_end_timer,
    history_window_ready,
    latest_history_ready,
    maybe_enable_headless,
    print_prediction,
    resize_for_display,
    resolve_existing_path,
    resolve_pose_backend,
    resolve_realtime_class_labels,
    resolve_runtime_pose_source,
    run_last13_history_prediction,
    run_temporal_prediction,
    trim_history_buffer,
    validate_args,
    vote_ranked_prediction,
    warmup_frame_count,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLIPGCN realtime inference from a desktop screen region.")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_ROOT / "config_final_aug.yaml"),
        help="Path to the CLIPGCN YAML config. It also defines the fusion head and the gating thresholds.",
    )
    parser.add_argument(
        "--class-split-dir",
        default=None,
        help="Directory whose metadata.json defines seen/unseen classes. Defaults to config data.train.data_dir.",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=["unseen", "seen", "all"],
        default="all",
        help="Which action text labels are valid predictions.",
    )
    parser.add_argument(
        "--class-config",
        default=None,
        help=(
            "Optional editable class-selection file (.csv/.tsv/.yaml/.json). "
            "Each class can set enabled true/false and split seen/unseen."
        ),
    )
    parser.add_argument(
        "--all-classes-seen",
        action="store_true",
        help="Override split metadata and treat every label from the xlsx/class config as seen with no unseen classes.",
    )
    parser.add_argument(
        "--seen-classes",
        default=None,
        help=(
            "Optional seen class override, e.g. '0-57' or 'A01,A02,A58'. "
            "Numeric IDs are zero-based; Axx IDs are converted to zero-based labels."
        ),
    )
    parser.add_argument(
        "--unseen-classes",
        default=None,
        help=(
            "Optional unseen class override, e.g. '9,10,11,17,49' or 'A10,A11,A12,A18,A50'. "
            "If only unseen is set, all other labels 0..54 become seen."
        ),
    )
    parser.add_argument(
        "--exclude-classes",
        default=None,
        help=(
            "Optional class IDs to remove from the candidate pool entirely, e.g. 'A01,A02' or '0,1'. "
            "Excluded labels are removed after seen/unseen overrides are applied."
        ),
    )
    parser.add_argument(
        "--unseen-score-scale",
        type=float,
        default=1.3,
        help="Multiplier applied to unseen class confidence scores before top-k prediction.",
    )
    parser.add_argument("--clipgcn-checkpoint", default=None, help="Optional CLIPGCN checkpoint override.")

    parser.add_argument(
        "--screen-region",
        default=None,
        help="Screen capture region as x,y,width,height. If omitted, captures the full display.",
    )
    parser.add_argument("--screen-left", type=int, default=0, help="Left edge of the capture region.")
    parser.add_argument("--screen-top", type=int, default=0, help="Top edge of the capture region.")
    parser.add_argument("--screen-width", type=int, default=None, help="Capture region width.")
    parser.add_argument("--screen-height", type=int, default=None, help="Capture region height.")
    parser.add_argument(
        "--screen-fps",
        type=float,
        default=30.0,
        help="Maximum screen capture FPS. Use 0 to disable throttling.",
    )

    parser.add_argument("--frames", type=int, default=13, help="Rolling frame window. Must match training.")
    parser.add_argument("--predict-every", type=int, default=13, help="Run recognition once every N captured frames.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--display-filter-window",
        type=int,
        default=10,
        help=(
            "Smooth displayed top-k labels by majority vote over the latest N predictions at each rank. "
            "Use 1 to show raw predictions."
        ),
    )
    # The same single decision policy used by webcam/ROS realtime. YAML holds
    # the defaults and these flags are optional runtime overrides.
    parser.add_argument(
        "--decision-entropy-threshold",
        type=float,
        default=None,
    )
    parser.add_argument("--decision-temperature", type=float, default=None)
    parser.add_argument(
        "--temporal-strategy",
        choices=["short", "short-long", "uniform3s", "last13"],
        default="short-long",
        help=(
            "short/short-long/uniform3s sample 13 frames from time windows. "
            "last13 uses the latest contiguous 13 captured frames like the old realtime demo."
        ),
    )
    parser.add_argument("--short-window-seconds", type=float, default=2.0, help="Seconds covered by 13 frames for S actions.")
    parser.add_argument("--long-window-seconds", type=float, default=4.0, help="Seconds covered by 13 frames for L actions.")
    parser.add_argument(
        "--uniform-window-seconds",
        type=float,
        default=3.0,
        help="Seconds covered by 13 frames when --temporal-strategy uniform3s is used.",
    )
    parser.add_argument(
        "--long-rerank-top-k",
        type=int,
        default=5,
        help="Run the long-window pass only if an L action appears in this many short-window candidates.",
    )
    parser.add_argument(
        "--pose-source",
        choices=["rtmpose", "mediapipe", "zero"],
        default=None,
        help=(
            "Realtime skeleton estimator. Defaults to whichever one the checkpoint's CTR-GCN "
            "backbone was trained on (rtmpose for COCO-17, mediapipe for NTU-25)."
        ),
    )
    parser.add_argument(
        "--rtmpose-mode",
        choices=["lightweight", "balanced", "performance"],
        default="balanced",
        help="rtmlib model pair. balanced is what the training extraction used.",
    )
    parser.add_argument("--rtmpose-backend", default="onnxruntime")
    parser.add_argument("--rtmpose-device", default=None, help="cuda or cpu. Defaults to cuda when available.")
    parser.add_argument("--mediapipe-model-complexity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--mediapipe-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--mediapipe-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--mediapipe-min-visibility", type=float, default=0.2)
    parser.add_argument(
        "--mediapipe-model-asset",
        default=default_mediapipe_model_asset(),
        help=(
            "Path to a MediaPipe Tasks pose_landmarker .task model. Required when the installed "
            "mediapipe package exposes tasks.vision.PoseLandmarker instead of the legacy solutions API. "
            "Can also be set with MEDIAPIPE_POSE_MODEL."
        ),
    )
    parser.add_argument("--window-name", default="CLIPGCN screen realtime")
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Rendered window width. Zero with --display-height 0 keeps the captured size.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=0,
        help="Rendered window height. Zero preserves the captured aspect ratio.",
    )
    parser.add_argument(
        "--hide-person-boxes",
        action="store_true",
        help="Compatibility flag shared with the webcam/ROS overlay.",
    )
    parser.add_argument("--headless", action="store_true", help="Print predictions without opening a display window.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback when the config requests CUDA.")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cudnn benchmark for fixed-size realtime inference on CUDA.",
    )

    parser.add_argument("--x3d-root", default=str(X3D_ROOT))
    parser.add_argument(
        "--x3d-config",
        default=str(X3D_ROOT / "configs" / "x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml"),
    )
    parser.add_argument(
        "--x3d-checkpoint",
        default=str(X3D_ROOT / "outputs" / "x3d-s_clipgcn_tensor_cs_70_10_20_182" / "model_007000.pth"),
    )
    parser.add_argument("--x3d-layer", default="s5")

    # Left unset on purpose: the skeleton backbone has to match the checkpoint's
    # joint count, so the defaults are filled in from the config in main().
    parser.add_argument("--ctrgcn-root", default=None)
    parser.add_argument("--ctrgcn-config", default=None)
    parser.add_argument("--ctrgcn-weights", default=None)
    parser.add_argument("--ctrgcn-hook-layer", default="l4")

    parser.add_argument("--yolo-repo", default=default_yolo_repo())
    parser.add_argument("--yolo-weights", default=default_yolo_weights())
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-half", action="store_true", help="Run YOLO in FP16 on CUDA.")
    parser.add_argument(
        "--yolo-detect-every",
        type=int,
        default=1,
        help="Run YOLO once every N predictions and reuse the previous object map in between.",
    )
    parser.add_argument("--no-yolo", action="store_true", help="Use zero object maps instead of YOLO.")
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)
    return parser.parse_args()


def parse_screen_bbox(args):
    if args.screen_region:
        parts = [part.strip() for part in args.screen_region.split(",")]
        if len(parts) != 4:
            raise ValueError("--screen-region must be formatted as x,y,width,height.")
        left, top, width, height = (int(part) for part in parts)
    elif args.screen_width is None and args.screen_height is None:
        return None
    elif args.screen_width is not None and args.screen_height is not None:
        left, top, width, height = args.screen_left, args.screen_top, args.screen_width, args.screen_height
    else:
        raise ValueError("--screen-width and --screen-height must be provided together.")

    if width <= 0 or height <= 0:
        raise ValueError("Screen capture width and height must be positive.")
    return (left, top, left + width, top + height)


class ScreenCaptureSource:
    """Small VideoCapture-like wrapper around PIL.ImageGrab."""

    def __init__(self, args):
        self.bbox = parse_screen_bbox(args)
        self.min_interval = 1.0 / args.screen_fps if args.screen_fps and args.screen_fps > 0 else 0.0
        self.last_capture_time = 0.0

    def read(self):
        if self.min_interval:
            now = time.perf_counter()
            delay = self.min_interval - (now - self.last_capture_time)
            if delay > 0:
                time.sleep(delay)
        self.last_capture_time = time.perf_counter()

        image = ImageGrab.grab(bbox=self.bbox)
        frame_rgb = np.asarray(image.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, np.ascontiguousarray(frame_bgr)

    def release(self):
        return None


def describe_screen_region(bbox):
    if bbox is None:
        return "full display"
    left, top, right, bottom = bbox
    return f"x={left}, y={top}, width={right - left}, height={bottom - top}"


def validate_screen_args(args):
    validate_args(args)
    parse_screen_bbox(args)
    if args.screen_fps < 0:
        raise ValueError("--screen-fps must be non-negative.")


def main():
    args = parse_args()
    validate_screen_args(args)
    args.action_duration_types = DEFAULT_ACTION_DURATION_TYPES
    args = maybe_enable_headless(args)
    # The config decides the skeleton layout, and the layout decides which
    # estimator is the right one, so it has to be read first.
    config_path = resolve_existing_path(args.config, label="config")
    config = load_config(config_path)
    args = resolve_pose_backend(args, config)
    args.runtime_pose_source = resolve_runtime_pose_source(args)
    args.label_display_names = load_action_display_names(config, config_path)
    args.x3d_root = resolve_existing_path(args.x3d_root, label="X3D root")
    args.x3d_config = resolve_existing_path(
        args.x3d_config,
        search_roots=[Path(args.x3d_root), Path(args.x3d_root).parent, SCRIPT_ROOT],
        label="X3D config",
    )
    args.x3d_checkpoint = resolve_existing_path(
        args.x3d_checkpoint,
        search_roots=[Path(args.x3d_root), Path(args.x3d_root).parent, SCRIPT_ROOT],
        label="X3D checkpoint",
    )
    # --pose-source zero never touches the skeleton backbone, so do not demand it.
    if args.runtime_pose_source in ("mediapipe", "rtmpose"):
        args.ctrgcn_root = resolve_existing_path(args.ctrgcn_root, label="CTR-GCN root")
        args.ctrgcn_config = resolve_existing_path(
            args.ctrgcn_config,
            search_roots=[Path(args.ctrgcn_root), Path(args.ctrgcn_root).parent, SCRIPT_ROOT],
            label="CTR-GCN config",
        )
        args.ctrgcn_weights = resolve_existing_path(
            args.ctrgcn_weights,
            search_roots=[Path(args.ctrgcn_root), Path(args.ctrgcn_root).parent, SCRIPT_ROOT],
            label="CTR-GCN weights",
        )
    if not args.no_yolo:
        try:
            args.yolo_repo = resolve_existing_path(args.yolo_repo, label="YOLO repo")
        except FileNotFoundError as exc:
            print(f"Warning: {exc}. Falling back to zero object maps.")
            args.no_yolo = True
        else:
            try:
                args.yolo_weights = resolve_existing_path(
                    args.yolo_weights,
                    search_roots=[Path(args.yolo_repo), Path(args.yolo_repo).parent, SCRIPT_ROOT],
                    label="YOLO weights",
                )
            except FileNotFoundError as exc:
                print(f"Warning: {exc}. Falling back to zero object maps.")
                args.no_yolo = True

    requested_device = str(config["runtime"].get("device"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "The config requests CUDA, but torch.cuda.is_available() is False. "
            "Pass --allow-cpu intentionally, or run in an environment with GPU access."
        )
    device = get_device(config["runtime"].get("device"))
    print_device_info(device)
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    class_split_dir = args.class_split_dir or get_path_from_config(config_path, config["data"]["train"]["data_dir"])
    class_split_dir = get_path_from_config(config_path, class_split_dir)
    class_selection = resolve_realtime_class_labels(args, class_split_dir, all_labels=args.label_display_names.keys())
    unseen_labels = class_selection["unseen_labels"]
    args.unseen_label_set = set(unseen_labels)
    args.class_score_adjustments = class_selection.get("score_adjustments", {})
    candidate_labels = class_selection["candidate_labels"]

    x3d_model, x3d_captured, x3d_hook, x3d_cfg = load_x3d_model(args, device)
    ctrgcn_model = None
    ctrgcn_captured = None
    ctrgcn_hook = None
    if args.runtime_pose_source in ("mediapipe", "rtmpose"):
        ctrgcn_model, ctrgcn_captured, ctrgcn_hook = load_ctrgcn_model(args, device)
    yolo_model = load_yolo_model(args, device)
    object_map_runner = ObjectMapRunner(yolo_model, device, args)
    clipgcn_model, checkpoint_path, candidate_labels = load_clipgcn_model(
        args,
        config,
        config_path,
        device,
        candidate_labels,
    )
    candidate_labels = [int(label) for label in candidate_labels]
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"
    args.decision_gate = build_decision_gate(config, candidate_labels, unseen_labels, args)
    # Both estimators are executed once per captured frame. RTMPose results are
    # cached on history entries so sampled prediction windows only pack them.
    pose_source = MediaPipePoseSource(args) if args.runtime_pose_source == "mediapipe" else None
    args.pose_estimator = RTMPoseSource(args) if args.runtime_pose_source == "rtmpose" else None
    cap = ScreenCaptureSource(args)
    print("Realtime CLIPGCN screen-region inference")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  class_split_dir: {class_split_dir}")
    print(f"  class_selection: {class_selection['source']}")
    print(f"  candidate_scope: {args.candidate_scope}")
    print(f"  candidate_labels: {candidate_labels}")
    print(f"  seen_labels: {class_selection['seen_labels']}")
    print(f"  unseen_labels: {unseen_labels}")
    if class_selection["excluded_labels"]:
        print(f"  excluded_labels: {class_selection['excluded_labels']}")
    if args.class_score_adjustments:
        print(f"  class_score_adjustments: {args.class_score_adjustments}")
    print(f"  unseen_score_scale: {args.unseen_score_scale:g}")
    print(f"  display_filter_window: {args.display_filter_window}")
    gate = args.decision_gate
    if gate.enabled:
        print(
            "  decision_gate: "
            f"YOLO person required; H<{gate.entropy_threshold:g} -> class; "
            f"otherwise ANALYSING"
        )
        print(f"  decision_temperature: {gate.temperature:g}")
    else:
        print("  decision_gate: disabled in YAML")
    print(f"  pose_source: {args.pose_source}")
    print(f"  pose_layout: {args.pose_layout} ({args.num_joints} joints)")
    if args.pose_estimator is not None:
        print(f"  rtmpose: mode={args.pose_estimator.mode} device={args.pose_estimator.device}")
    print(f"  runtime_pose_source: {args.runtime_pose_source}")
    print(f"  temporal_strategy: {args.temporal_strategy}")
    if args.temporal_strategy == "last13":
        print(f"  latest_contiguous_frames: {args.frames}")
    elif args.temporal_strategy == "uniform3s":
        print(f"  uniform_window_seconds: {args.uniform_window_seconds:g}")
    else:
        print(f"  short_window_seconds: {args.short_window_seconds:g}")
        print(f"  long_window_seconds: {args.long_window_seconds:g}")
    print(f"  screen_region: {describe_screen_region(cap.bbox)}")
    print(f"  screen_fps: {args.screen_fps:g}")
    print("  YOLO overlay targets: stove, biscuits, pot")
    print("  latency: E2E=newest raw frame to voted prediction; core=model section only")
    print("  quit: press q or ESC")

    history_buffer = deque()
    prediction = None
    display_prediction = None
    prediction_history = deque(maxlen=args.display_filter_window)
    captured_frames = 0
    display_window_initialized = False
    current_person_detections = []

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the screen.")

            # Screen-grab waiting is excluded; raw-frame processing through the
            # final voted prediction is included in the E2E latency.
            frame_processing_start = time.perf_counter()
            captured_frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = time.monotonic()
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                current_object_maps = object_map_runner([frame_rgb])
            current_person_detections = [
                dict(item) for item in object_map_runner.last_person_detections
            ]
            current_watched_detections = [
                dict(item) for item in object_map_runner.last_watched_detections
            ]
            args.current_person_present = bool(current_person_detections)
            skeleton_sample = None
            if pose_source is not None:
                skeleton_sample = pose_source.process(frame_rgb)
            entry = {
                "timestamp": timestamp,
                "frame_rgb": frame_rgb,
                "skeleton": skeleton_sample,
                "object_map": current_object_maps[0].detach(),
                "yolo_watched_detections": current_watched_detections,
                "yolo_person_detections": current_person_detections,
            }
            if args.pose_estimator is not None:
                entry["rtmpose"] = args.pose_estimator.keypoints(frame_rgb)
            history_buffer.append(entry)
            trim_history_buffer(history_buffer, args, timestamp)

            if args.temporal_strategy == "last13":
                frame_count = min(len(history_buffer), args.frames)
                should_predict = (
                    latest_history_ready(history_buffer, args)
                    and captured_frames % args.predict_every == 0
                )
            else:
                frame_count = warmup_frame_count(history_buffer, args)
                pose_ready = args.runtime_pose_source != "mediapipe" or all(
                    sample.get("skeleton") is not None for sample in history_buffer
                )
                should_predict = (
                    history_window_ready(history_buffer, active_window_seconds(args), args)
                    and pose_ready
                    and captured_frames % args.predict_every == 0
                )
            if should_predict:
                if args.temporal_strategy == "last13":
                    prediction = run_last13_history_prediction(
                        history_buffer,
                        args,
                        device,
                        x3d_cfg,
                        x3d_model,
                        x3d_captured,
                        ctrgcn_model,
                        ctrgcn_captured,
                        object_map_runner,
                        clipgcn_model,
                        candidate_labels,
                        unseen_labels,
                        use_amp,
                    )
                else:
                    prediction = run_temporal_prediction(
                        history_buffer,
                        args,
                        device,
                        x3d_cfg,
                        x3d_model,
                        x3d_captured,
                        ctrgcn_model,
                        ctrgcn_captured,
                        object_map_runner,
                        clipgcn_model,
                        candidate_labels,
                        unseen_labels,
                        use_amp,
                    )
                prediction_history.append(prediction)
                display_prediction = vote_ranked_prediction(prediction_history, args)
                end_to_end_elapsed = finish_end_to_end_timer(frame_processing_start, device)
                prediction["end_to_end_elapsed"] = end_to_end_elapsed
                display_prediction["end_to_end_elapsed"] = end_to_end_elapsed
                if args.headless:
                    print_prediction(display_prediction, args)

            if not args.headless:
                display_frame = resize_for_display(frame_bgr, args)
                draw_overlay(
                    display_frame,
                    display_prediction,
                    frame_count,
                    args,
                    person_detections=current_person_detections,
                    source_frame_shape=frame_bgr.shape,
                )
                if not display_window_initialized:
                    cv2.namedWindow(
                        args.window_name,
                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
                    )
                    cv2.resizeWindow(
                        args.window_name,
                        display_frame.shape[1],
                        display_frame.shape[0],
                    )
                    display_window_initialized = True
                cv2.imshow(args.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        x3d_hook.remove()
        if ctrgcn_hook is not None:
            ctrgcn_hook.remove()
        if pose_source is not None:
            pose_source.close()
        if getattr(args, "pose_estimator", None) is not None:
            args.pose_estimator.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

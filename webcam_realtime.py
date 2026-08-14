#!/usr/bin/env python
"""Realtime webcam inference for CLIPGCN.

The trained CLIPGCN model expects pre-extracted video, pose, object and joint
features. This script reads frames from a local webcam, then runs X3D, YOLO,
MediaPipe/CTR-GCN, and CLIPGCN online.
"""

import argparse
import csv
import importlib.util
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from action_label_utils import load_action_display_names
from gating import ANALYSING_LABEL, UNKNOWN_LABEL, PersonEntropyDecisionGate
from test import (
    apply_unseen_score_scale,
    load_split_classes,
    load_split_metadata,
    logits_to_cosine,
    logits_to_unit_cosine_scores,
)
from test_raw_end_to_end import (
    CTRGCN_ROOT,
    X3D_ROOT,
    ObjectMapRunner,
    ctrgcn_pose_from_model,
    load_clipgcn_model,
    load_ctrgcn_model,
    load_x3d_model,
    load_yolo_model,
    x3d_features_from_model,
    x3d_tensor_from_frames,
)
from train import get_device, get_path_from_config, load_config, print_device_info


SCRIPT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent
# The skeleton layout is not a free choice: it is baked into the checkpoint by
# model.fusion.num_joints (33 video channels + 17 joints, or 25 + 25).
CTRGCN_17_ROOT = WORKSPACE_ROOT / "CTR-GCN_17"
POSE_BACKENDS = {
    25: {
        "layout": "ntu25",
        # The ETRI backbone was trained on 13-frame Kinect clips, so the window
        # goes in as captured.
        "source": "mediapipe",
        "window_size": None,
        "root": CTRGCN_ROOT,
        "config": CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "config.yaml",
        "weights": CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "runs-50-2700.pt",
    },
    17: {
        "layout": "coco17",
        # This backbone was trained on RTMPose keypoints, and its feeder
        # upsamples the 13 stored frames to 64 before the multi-scale TCN.
        "source": "rtmpose",
        "window_size": 64,
        "root": CTRGCN_17_ROOT,
        "config": CTRGCN_17_ROOT / "config" / "etri-coco17" / "ctrgcn_joint_coco17_13.yaml",
        "weights": CTRGCN_17_ROOT / "runs-50-600.pt",
    },
}

# COCO-17 keypoint order, expressed as MediaPipe Pose landmark indices. Only
# used by the --pose-source mediapipe fallback; the trained pose stream comes
# from RTMPose.
COCO17_FROM_MEDIAPIPE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# extract_rtmpose_coco17.py keeps the two most confident bodies per frame.
RTMPOSE_MAX_PERSONS = 2

DEFAULT_ACTION_DURATION_TYPES = {
    0: "L",
    1: "L",
    2: "L",
    3: "L",
    4: "L",
    5: "S",
    6: "L",
    7: "L",
    8: "S",
    9: "S",
    10: "S",
    11: "S",
    12: "S",
    13: "L",
    14: "L",
    15: "S",
    16: "S",
    17: "L",
    18: "L",
    19: "L",
    20: "L",
    21: "L",
    22: "S",
    23: "S",
    24: "S",
    25: "S",
    26: "L",
    27: "S",
    28: "L",
    29: "L",
    30: "S",
    31: "S",
    32: "S",
    33: "S",
    34: "S",
    35: "S",
    36: "S",
    37: "S",
    38: "S",
    39: "S",
    40: "S",
    41: "S",
    42: "S",
    43: "S",
    44: "S",
    45: "S",
    46: "L",
    47: "S",
    48: "S",
    49: "S",
    50: "S",
    51: "L",
    52: "S",
    53: "L",
    54: "L",
}


def resolve_existing_path(path_value, *, search_roots=None, label="path"):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return str(path)

    roots = []
    if search_roots:
        roots.extend(Path(root) for root in search_roots)
    roots.extend([Path.cwd(), SCRIPT_ROOT, WORKSPACE_ROOT])

    seen = set()
    for root in roots:
        candidate = (root / path).resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate)

    tried = ", ".join(str((root / path).resolve()) for root in roots)
    raise FileNotFoundError(f"{label} not found: {path}. Tried: {tried}")


def default_yolo_repo():
    candidates = [
        WORKSPACE_ROOT / "yolov5",
        SCRIPT_ROOT / "yolov5",
        Path("/workspace/yolov5"),
    ]
    for candidate in candidates:
        if (candidate / "hubconf.py").exists():
            return str(candidate)
    return str(candidates[0])


def default_yolo_weights():
    candidates = [
        SCRIPT_ROOT / "local_models" / "yolov5m.pt",
        WORKSPACE_ROOT / "yolov5" / "yolov5m.pt",
        SCRIPT_ROOT / "yolov5" / "yolov5m.pt",
        Path("/workspace/yolov5/yolov5m.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def default_mediapipe_model_asset():
    env_path = os.environ.get("MEDIAPIPE_POSE_MODEL")
    if env_path and Path(env_path).expanduser().exists():
        return env_path

    names = [
        "pose_landmarker_full.task",
        "pose_landmarker_lite.task",
        "pose_landmarker_heavy.task",
    ]
    roots = [SCRIPT_ROOT / "local_models", SCRIPT_ROOT, WORKSPACE_ROOT, Path("/workspace"), Path.cwd()]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
    return env_path


def build_arg_parser(description="Run CLIPGCN realtime inference from a local webcam."):
    parser = argparse.ArgumentParser(description=description)
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
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
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
    # One decision policy only: YOLO person first, then entropy. Defaults come
    # from inference.decision_gate in YAML.
    parser.add_argument(
        "--decision-entropy-threshold",
        type=float,
        default=None,
        help="Show a class only when a YOLO person exists and H is below this value.",
    )
    parser.add_argument(
        "--decision-temperature",
        type=float,
        default=None,
        help="Override softmax temperature used only to calculate H.",
    )
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
            "backbone was trained on (rtmpose for COCO-17, mediapipe for NTU-25). "
            "Use zero only for debugging."
        ),
    )
    parser.add_argument(
        "--rtmpose-mode",
        choices=["lightweight", "balanced", "performance"],
        default="balanced",
        help="rtmlib model pair. balanced is what the training extraction used.",
    )
    parser.add_argument("--rtmpose-backend", default="onnxruntime")
    parser.add_argument(
        "--rtmpose-device",
        default=None,
        help="cuda or cpu. Defaults to cuda when onnxruntime exposes a CUDA provider.",
    )
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
    parser.add_argument("--window-name", default="CLIPGCN realtime")
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Display width in pixels. Use 0 together with --display-height 0 for the native camera size.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=0,
        help="Display height in pixels. The default 0 preserves the camera aspect ratio.",
    )
    parser.add_argument("--headless", action="store_true", help="Print predictions without opening a display window.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback when the config requests CUDA.")
    parser.add_argument(
        "--runtime-device",
        "--device",
        dest="runtime_device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help=(
            "Override the whole realtime pipeline device. cpu forces CLIPGCN, X3D, "
            "CTR-GCN, YOLO, and RTMPose onto CPU; cuda requires CUDA for both "
            "PyTorch and RTMPose; auto selects the best available device for each. "
            "When omitted, the YAML runtime.device and --rtmpose-device behavior are preserved."
        ),
    )
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
        help="Run YOLO once every N captured frames and reuse its map/boxes in between.",
    )
    parser.add_argument("--no-yolo", action="store_true", help="Use zero object maps instead of YOLO.")
    parser.add_argument(
        "--hide-person-boxes",
        action="store_true",
        help="Do not draw YOLO person boxes in the realtime display.",
    )
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)


def validate_args(args):
    if args.frames != 13:
        raise ValueError(
            "This CLIPGCN checkpoint expects 13-frame features. "
            "Keep --frames 13 unless you retrain/update the fusion model."
        )
    if args.predict_every <= 0:
        raise ValueError("--predict-every must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.display_filter_window <= 0:
        raise ValueError("--display-filter-window must be positive.")
    if getattr(args, "display_width", 0) < 0:
        raise ValueError("--display-width must be non-negative.")
    if getattr(args, "display_height", 0) < 0:
        raise ValueError("--display-height must be non-negative.")
    if args.yolo_detect_every <= 0:
        raise ValueError("--yolo-detect-every must be positive.")
    decision_entropy_threshold = getattr(args, "decision_entropy_threshold", None)
    if decision_entropy_threshold is not None and not 0.0 <= decision_entropy_threshold <= 1.0:
        raise ValueError("--decision-entropy-threshold must be in [0, 1].")
    decision_temperature = getattr(args, "decision_temperature", None)
    if decision_temperature is not None and decision_temperature <= 0:
        raise ValueError("--decision-temperature must be positive.")
    if args.unseen_score_scale <= 0:
        raise ValueError("--unseen-score-scale must be positive.")
    temporal_strategy = getattr(args, "temporal_strategy", None)
    if temporal_strategy == "uniform3s":
        if getattr(args, "uniform_window_seconds", 0) <= 0:
            raise ValueError("--uniform-window-seconds must be positive.")
    elif temporal_strategy != "last13":
        if getattr(args, "short_window_seconds", 0) <= 0:
            raise ValueError("--short-window-seconds must be positive.")
        if getattr(args, "long_window_seconds", 0) <= 0:
            raise ValueError("--long-window-seconds must be positive.")
        if getattr(args, "long_window_seconds", 0) < getattr(args, "short_window_seconds", 0):
            raise ValueError("--long-window-seconds must be greater than or equal to --short-window-seconds.")
        if getattr(args, "long_rerank_top_k", 0) <= 0:
            raise ValueError("--long-rerank-top-k must be positive.")


def resolve_pose_backend(args, config):
    """Pick the skeleton layout and CTR-GCN backbone the checkpoint was built on.

    The fusion head concatenates video and pose channels into a fixed 50, so a
    33-channel video projection only leaves room for 17 joints. Feeding it the
    NTU-25 stream would not just hurt accuracy, it would not even run.
    """

    fusion_config = (config.get("model") or {}).get("fusion") or {}
    num_joints = int(fusion_config.get("num_joints", 25))
    if num_joints not in POSE_BACKENDS:
        raise ValueError(
            f"model.fusion.num_joints={num_joints} has no realtime skeleton backend. "
            f"Supported: {sorted(POSE_BACKENDS)}"
        )

    backend = POSE_BACKENDS[num_joints]
    args.num_joints = num_joints
    args.pose_layout = backend["layout"]
    args.ctrgcn_window_size = backend["window_size"]
    for name in ("root", "config", "weights"):
        attribute = f"ctrgcn_{name}"
        if getattr(args, attribute, None) is None:
            setattr(args, attribute, str(backend[name]))

    if getattr(args, "pose_source", None) is None:
        args.pose_source = backend["source"]
    elif args.pose_source not in ("zero", backend["source"]):
        # Estimators are not interchangeable: the backbone learned one
        # estimator's keypoint conventions and its confidence channel.
        print(
            f"Warning: this checkpoint's pose stream was trained on {backend['source']}, "
            f"but --pose-source {args.pose_source} was requested. Expect degraded accuracy."
        )
    return args


def build_decision_gate(config, candidate_labels, unseen_labels, args):
    """Build the only realtime decision policy, with optional CLI overrides."""

    gate_config = dict((config.get("inference") or {}).get("decision_gate") or {})
    override_names = {
        "decision_entropy_threshold": "entropy_threshold",
        "decision_temperature": "temperature",
    }
    for argument_name, config_name in override_names.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            gate_config[config_name] = value
    return PersonEntropyDecisionGate(candidate_labels, unseen_labels, gate_config)


def parse_class_spec(spec, *, label_min=0, label_max=54):
    if spec is None:
        return None

    value = str(spec).strip()
    if value == "" or value.lower() in {"none", "empty", "[]"}:
        return []

    labels = []
    for raw_token in value.replace(";", ",").replace(" ", ",").split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_token, end_token = [part.strip() for part in token.split("-", 1)]
            start = parse_single_class_id(start_token)
            end = parse_single_class_id(end_token)
            if end < start:
                raise ValueError(f"Invalid descending class range: {token}")
            labels.extend(range(start, end + 1))
        else:
            labels.append(parse_single_class_id(token))

    unique_labels = sorted(set(labels))
    invalid = [label for label in unique_labels if label < label_min or label > label_max]
    if invalid:
        raise ValueError(
            f"Class IDs must be in {label_min}..{label_max} or A{label_min + 1:02d}..A{label_max + 1:02d}. "
            f"Invalid: {invalid}"
        )
    return unique_labels


def parse_single_class_id(token):
    token = str(token).strip()
    if not token:
        raise ValueError("Empty class ID in class list.")
    if token[0].lower() == "a":
        return int(token[1:]) - 1
    return int(token)


def parse_config_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y", "on", "enable", "enabled", "active"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disable", "disabled", "inactive"}:
        return False
    raise ValueError(f"Invalid boolean value in class config: {value!r}")


def parse_config_float(value, default=0.0):
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    return float(text)


def normalize_class_split(value, default="seen"):
    if value is None:
        return default
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text == "":
        return default
    if text in {"seen", "s", "train", "trained"}:
        return "seen"
    if text in {"unseen", "u", "zsl", "holdout", "held_out", "heldout"}:
        return "unseen"
    if text in {"disabled", "disable", "excluded", "exclude", "remove", "removed", "ignore", "ignored", "off"}:
        return "disabled"
    raise ValueError(f"Invalid split value in class config: {value!r}")


def first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] not in {None, ""}:
            return mapping[key]
    return default


def load_class_config(path):
    resolved_path = resolve_existing_path(path, search_roots=[SCRIPT_ROOT, WORKSPACE_ROOT], label="class config")
    suffix = Path(resolved_path).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(resolved_path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
        return {"path": resolved_path, "classes": rows}

    with open(resolved_path, "r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(handle) or {}
        else:
            raise ValueError("Class config must be .csv, .tsv, .yaml, .yml, or .json.")

    if isinstance(data, list):
        data = {"classes": data}
    if not isinstance(data, dict):
        raise ValueError("YAML/JSON class config must be a mapping or a list of class records.")
    data["path"] = resolved_path
    return data


def iter_class_config_records(classes):
    if classes is None:
        return []
    if isinstance(classes, dict):
        records = []
        for class_id, value in classes.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("id", class_id)
            else:
                record = {"id": class_id, "split": value}
            records.append(record)
        return records
    if isinstance(classes, list):
        records = []
        for item in classes:
            if isinstance(item, dict):
                records.append(dict(item))
            else:
                records.append({"id": item})
        return records
    raise ValueError("Class config 'classes' must be a list or mapping.")


def labels_from_class_config(path, universe):
    config = load_class_config(path)
    default_enabled = parse_config_bool(first_present(config, ["default_enabled", "enabled"], True), default=True)
    default_split = normalize_class_split(first_present(config, ["default_split", "split"], "seen"), default="seen")
    if default_split == "disabled":
        default_enabled = False
        default_split = "seen"

    state = {
        int(label): {
            "enabled": default_enabled,
            "split": default_split,
            "score_scale": 1.0,
            "score_bias": 0.0,
        }
        for label in universe
    }

    for record in iter_class_config_records(config.get("classes")):
        class_id = first_present(record, ["id", "ID", "action_id", "class_id", "label", "label_id"])
        if class_id is None:
            raise ValueError(f"Class config record is missing an id/label field: {record}")

        class_id_text = str(class_id).strip()
        if class_id_text.upper() in {"DEFAULT", "*", "ALL"}:
            default_enabled = parse_config_bool(
                first_present(record, ["enabled", "enable", "active"], default_enabled),
                default=default_enabled,
            )
            default_split = normalize_class_split(
                first_present(record, ["split", "status", "type"], default_split),
                default=default_split,
            )
            if default_split == "disabled":
                default_enabled = False
                default_split = "seen"
            for label in universe:
                current = state[int(label)]
                state[int(label)] = {
                    "enabled": default_enabled,
                    "split": default_split,
                    "score_scale": parse_config_float(
                        first_present(record, ["score_scale", "scale"], current["score_scale"]),
                        default=current["score_scale"],
                    ),
                    "score_bias": parse_config_float(
                        first_present(record, ["score_bias", "bias"], current["score_bias"]),
                        default=current["score_bias"],
                    )
                    - parse_config_float(first_present(record, ["score_penalty", "penalty"], 0.0), default=0.0),
                }
            continue

        label = parse_single_class_id(class_id_text)
        if label not in state:
            raise ValueError(f"Class config label out of range: {class_id}")

        current = state[label]
        split = normalize_class_split(first_present(record, ["split", "status", "type"], current["split"]), default=current["split"])
        enabled = parse_config_bool(
            first_present(record, ["enabled", "enable", "active"], current["enabled"]),
            default=current["enabled"],
        )
        score_scale = parse_config_float(
            first_present(record, ["score_scale", "scale"], current["score_scale"]),
            default=current["score_scale"],
        )
        score_bias = parse_config_float(
            first_present(record, ["score_bias", "bias"], current["score_bias"]),
            default=current["score_bias"],
        ) - parse_config_float(first_present(record, ["score_penalty", "penalty"], 0.0), default=0.0)
        if split == "disabled":
            enabled = False
            split = current["split"]
        state[label] = {
            "enabled": enabled,
            "split": split,
            "score_scale": score_scale,
            "score_bias": score_bias,
        }

    seen_labels = []
    unseen_labels = []
    excluded_labels = []
    score_adjustments = {}
    for label in sorted(state):
        item = state[label]
        if not item["enabled"]:
            excluded_labels.append(label)
        elif item["split"] == "unseen":
            unseen_labels.append(label)
        else:
            seen_labels.append(label)
        if item["score_scale"] != 1.0 or item["score_bias"] != 0.0:
            score_adjustments[label] = {
                "scale": float(item["score_scale"]),
                "bias": float(item["score_bias"]),
            }

    return {
        "seen_labels": seen_labels,
        "unseen_labels": unseen_labels,
        "excluded_labels": excluded_labels,
        "score_adjustments": score_adjustments,
        "path": config["path"],
    }


def select_candidate_labels(seen_labels, unseen_labels, candidate_scope):
    if candidate_scope == "seen":
        return sorted(seen_labels)
    if candidate_scope == "unseen":
        return sorted(unseen_labels)
    if candidate_scope == "all":
        return sorted(set(seen_labels) | set(unseen_labels))
    raise ValueError(f"Unsupported candidate scope: {candidate_scope}")


def resolve_realtime_class_labels(args, class_split_dir, *, all_labels=range(55)):
    all_labels = sorted(int(label) for label in all_labels)
    metadata = load_split_metadata(class_split_dir)
    metadata_seen = [int(label) for label in metadata.get("seen_classes", all_labels)]
    metadata_unseen = [int(label) for label in metadata.get("unseen_classes", [])]
    universe = sorted(set(metadata_seen) | set(metadata_unseen) | set(all_labels))
    label_min = min(universe)
    label_max = max(universe)

    seen_override = parse_class_spec(getattr(args, "seen_classes", None), label_min=label_min, label_max=label_max)
    unseen_override = parse_class_spec(getattr(args, "unseen_classes", None), label_min=label_min, label_max=label_max)
    cli_exclude_labels = set(
        parse_class_spec(getattr(args, "exclude_classes", None), label_min=label_min, label_max=label_max) or []
    )
    class_config_path = getattr(args, "class_config", None)
    class_config_selection = labels_from_class_config(class_config_path, universe) if class_config_path else None

    if getattr(args, "all_classes_seen", False):
        seen_labels = list(universe)
        unseen_labels = []
        base_excluded_labels = set()
        score_adjustments = {}
    elif seen_override is None and unseen_override is None:
        if class_config_selection is not None:
            seen_labels = class_config_selection["seen_labels"]
            unseen_labels = class_config_selection["unseen_labels"]
            base_excluded_labels = set(class_config_selection["excluded_labels"])
            score_adjustments = dict(class_config_selection.get("score_adjustments", {}))
        else:
            seen_labels = metadata_seen
            unseen_labels = metadata_unseen
            base_excluded_labels = set()
            score_adjustments = {}
    elif seen_override is None:
        unseen_set = set(unseen_override)
        seen_labels = [label for label in universe if label not in unseen_set]
        unseen_labels = list(unseen_override)
        base_excluded_labels = set()
        score_adjustments = {}
    elif unseen_override is None:
        seen_set = set(seen_override)
        seen_labels = list(seen_override)
        unseen_labels = [label for label in universe if label not in seen_set]
        base_excluded_labels = set()
        score_adjustments = {}
    else:
        seen_labels = list(seen_override)
        unseen_labels = list(unseen_override)
        base_excluded_labels = set()
        score_adjustments = {}

    overlap = sorted(set(seen_labels) & set(unseen_labels))
    if overlap:
        raise ValueError(f"Seen and unseen classes overlap: {overlap}")

    exclude_labels = base_excluded_labels | cli_exclude_labels
    if exclude_labels:
        seen_labels = [label for label in seen_labels if label not in exclude_labels]
        unseen_labels = [label for label in unseen_labels if label not in exclude_labels]
        score_adjustments = {
            label: value for label, value in score_adjustments.items() if label not in exclude_labels
        }

    candidate_labels = select_candidate_labels(seen_labels, unseen_labels, args.candidate_scope)
    if not candidate_labels:
        raise ValueError(
            "No candidate classes remain after applying --candidate-scope, --seen-classes, "
            "--unseen-classes, and --exclude-classes."
        )

    class_selection = {
        "seen_labels": sorted(seen_labels),
        "unseen_labels": sorted(unseen_labels),
        "excluded_labels": sorted(exclude_labels),
        "score_adjustments": score_adjustments,
        "candidate_labels": candidate_labels,
        "source": class_config_selection["path"]
        if class_config_selection is not None
        and not getattr(args, "all_classes_seen", False)
        and seen_override is None
        and unseen_override is None
        else (
            "override"
            if getattr(args, "all_classes_seen", False)
            or seen_override is not None
            or unseen_override is not None
            or cli_exclude_labels
            else "metadata"
        ),
    }
    return class_selection


def open_camera(args):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    return cap


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def maybe_enable_headless(args):
    if args.headless or has_display():
        return args
    print("Warning: no GUI display detected, enabling --headless automatically.")
    args.headless = True
    return args


def import_mediapipe_solutions_pose_module():
    try:
        from mediapipe.python.solutions import pose as mp_pose

        return mp_pose
    except Exception:
        import mediapipe as mp

        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            raise AttributeError("module 'mediapipe' has no attribute 'solutions'")
        return solutions.pose


def import_mediapipe_tasks_pose_api():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    if not hasattr(vision, "PoseLandmarker"):
        raise AttributeError("mediapipe.tasks.python.vision has no PoseLandmarker")
    return mp, mp_tasks, vision


def resolve_mediapipe_model_asset(args):
    path = args.mediapipe_model_asset or os.environ.get("MEDIAPIPE_POSE_MODEL")
    if not path:
        return None
    return resolve_existing_path(
        path,
        search_roots=[SCRIPT_ROOT, WORKSPACE_ROOT, Path("/workspace"), Path.cwd()],
        label="MediaPipe pose model asset",
    )


def mediapipe_diagnostics():
    lines = []
    spec = importlib.util.find_spec("mediapipe")
    if spec is None:
        return "mediapipe import spec: not found"

    lines.append(f"mediapipe import origin: {spec.origin}")
    lines.append(f"mediapipe search locations: {list(spec.submodule_search_locations or [])}")
    try:
        import mediapipe as mp

        lines.append(f"mediapipe module file: {getattr(mp, '__file__', None)}")
        lines.append(f"mediapipe has solutions: {hasattr(mp, 'solutions')}")
        lines.append(f"mediapipe has tasks: {hasattr(mp, 'tasks')}")
        lines.append(f"mediapipe sample attrs: {sorted(name for name in dir(mp) if not name.startswith('_'))[:30]}")
    except Exception as exc:
        lines.append(f"mediapipe import error: {exc}")
    return "\n".join(lines)


def resolve_runtime_pose_source(args):
    if args.pose_source == "rtmpose":
        try:
            import rtmlib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--pose-source rtmpose needs rtmlib. Install it with "
                "'pip install rtmlib onnxruntime-gpu', or run with --pose-source zero."
            ) from exc
        return "rtmpose"
    if args.pose_source != "mediapipe":
        return args.pose_source
    try:
        import_mediapipe_solutions_pose_module()
        return "mediapipe"
    except Exception as solutions_exc:
        try:
            import_mediapipe_tasks_pose_api()
            model_asset_path = resolve_mediapipe_model_asset(args)
            if model_asset_path is None:
                raise FileNotFoundError(
                    "MediaPipe Tasks PoseLandmarker requires a local pose_landmarker .task model. "
                    "Pass --mediapipe-model-asset or set MEDIAPIPE_POSE_MODEL."
                )
            return "mediapipe"
        except Exception as tasks_exc:
            model_asset_hint = (
                "Set --mediapipe-model-asset /path/to/pose_landmarker_full.task "
                "or MEDIAPIPE_POSE_MODEL when using the MediaPipe Tasks-only package."
            )
            if not (args.mediapipe_model_asset or os.environ.get("MEDIAPIPE_POSE_MODEL")):
                model_asset_hint = (
                    "The installed mediapipe package exposes the Tasks API, which requires a local "
                    "pose_landmarker .task model file. " + model_asset_hint
                )
            raise RuntimeError(
                "MediaPipe Pose API is required because --pose-source mediapipe was requested, "
                "but it could not be loaded.\n"
                f"Legacy solutions error: {solutions_exc}\n"
                f"Tasks PoseLandmarker error: {tasks_exc}\n"
                f"{model_asset_hint}\n"
                f"{mediapipe_diagnostics()}"
            ) from tasks_exc


def resolve_rtmpose_device(args):
    if args.rtmpose_device:
        return args.rtmpose_device
    try:
        import onnxruntime
    except ImportError:
        return "cpu"
    return "cuda" if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else "cpu"


def rtmpose_cuda_available():
    """Return whether ONNX Runtime can actually expose RTMPose's CUDA backend."""

    try:
        import onnxruntime
    except ImportError:
        return False
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def apply_runtime_device_override(args, config):
    """Apply one CLI device choice consistently across all realtime backends."""

    requested = getattr(args, "runtime_device", None)
    if requested is None:
        return

    runtime_config = config.setdefault("runtime", {})
    uses_rtmpose = getattr(args, "runtime_pose_source", None) == "rtmpose"

    if requested == "cpu":
        torch_device = "cpu"
        rtmpose_device = "cpu"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--runtime-device cuda was requested, but PyTorch cannot access CUDA. "
                "Use --runtime-device cpu or check the container GPU configuration."
            )
        if uses_rtmpose and not rtmpose_cuda_available():
            raise RuntimeError(
                "--runtime-device cuda was requested, but ONNX Runtime does not expose "
                "CUDAExecutionProvider for RTMPose. Install/use onnxruntime-gpu, or run "
                "with --runtime-device cpu/auto."
            )
        torch_device = "cuda:0"
        rtmpose_device = "cuda"
    else:  # auto
        torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        rtmpose_device = "cuda" if rtmpose_cuda_available() else "cpu"

    runtime_config["device"] = torch_device
    if uses_rtmpose:
        args.rtmpose_device = rtmpose_device

    if torch_device == "cpu":
        if args.yolo_half:
            print("Runtime device is CPU; disabling --yolo-half (FP16 CUDA only).")
        if args.cudnn_benchmark:
            print("Runtime device is CPU; disabling --cudnn-benchmark (CUDA only).")
        args.yolo_half = False
        args.cudnn_benchmark = False

    pose_text = rtmpose_device if uses_rtmpose else "not used"
    print(
        f"Runtime device override: {requested} "
        f"(PyTorch={torch_device}, RTMPose={pose_text})"
    )


def top_persons(keypoints, scores, max_persons=RTMPOSE_MAX_PERSONS):
    """Keep the most confident bodies; ETRI classes 44-47 are two-person."""

    keypoints = np.asarray(keypoints)
    if keypoints.ndim != 3 or len(keypoints) == 0:
        return None, None
    scores = np.asarray(scores)
    order = np.argsort(scores.sum(axis=1))[::-1][:max_persons]
    return keypoints[order], scores[order]


class RTMPoseSource:
    """RTMPose COCO-17 keypoints, the skeletons the backbone was trained on.

    Realtime extraction runs once per captured frame and stores the result on
    the history entry. Prediction windows then select and pack cached results.
    """

    def __init__(self, args):
        try:
            from rtmlib import Body
        except ImportError as exc:
            raise ImportError(
                "rtmlib is required for --pose-source rtmpose. Install it with "
                "'pip install rtmlib onnxruntime-gpu', or fall back to --pose-source zero."
            ) from exc

        self.device = resolve_rtmpose_device(args)
        self.mode = args.rtmpose_mode
        self.body = Body(mode=self.mode, backend=args.rtmpose_backend, device=self.device)
        self.max_persons = RTMPOSE_MAX_PERSONS

    def close(self):
        pass

    def keypoints(self, frame_rgb):
        # The extraction fed cv2 BGR frames, so match that channel order.
        detections = self.body(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        return top_persons(*detections, self.max_persons)

    def skeleton_from_frames(self, frames_rgb, samples=None):
        """Return [3, T, 17, 2] with channels (x, y, score).

        x/y are normalised to [-1, 1] over the frame, and the third channel is
        RTMPose's keypoint confidence - not a depth value. Frames where nobody
        was found stay zero, which is what the training extractor stored too.

        ``samples`` are the history entries the frames came from. Overlapping
        windows keep re-selecting the same frames, so the estimator output is
        cached on the entry and the short/long passes mostly hit that cache.
        """

        frames = list(frames_rgb)
        height, width = frames[0].shape[:2]
        clip = np.zeros((3, len(frames), 17, self.max_persons), dtype=np.float32)
        detected = 0
        for index, frame_rgb in enumerate(frames):
            sample = samples[index] if samples is not None else None
            if sample is not None and "rtmpose" in sample:
                keypoints, scores = sample["rtmpose"]
            else:
                keypoints, scores = self.keypoints(frame_rgb)
                if sample is not None:
                    sample["rtmpose"] = (keypoints, scores)
            if keypoints is None:
                continue
            detected += 1
            for person in range(len(keypoints)):
                clip[0, index, :, person] = keypoints[person, :, 0] / width * 2.0 - 1.0
                clip[1, index, :, person] = keypoints[person, :, 1] / height * 2.0 - 1.0
                clip[2, index, :, person] = scores[person]
        return clip, detected


class MediaPipePoseSource:
    """Converts MediaPipe's 33 landmarks into the layout the checkpoint expects.

    ``coco17`` mirrors the RTMPose skeleton the model was trained on; ``ntu25``
    is the older Kinect layout kept for the 25-joint checkpoints.
    """

    def __init__(self, args):
        self.num_joints = int(getattr(args, "num_joints", 25))
        self.landmarks_to_joints = (
            mediapipe_landmarks_to_coco17 if self.num_joints == 17 else mediapipe_landmarks_to_ntu25
        )
        self.min_visibility = float(args.mediapipe_min_visibility)
        self.backend = None
        self.mp = None
        self.last_timestamp_ms = 0
        try:
            mp_pose = import_mediapipe_solutions_pose_module()
            self.backend = "solutions"
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=args.mediapipe_model_complexity,
                enable_segmentation=False,
                min_detection_confidence=args.mediapipe_min_detection_confidence,
                min_tracking_confidence=args.mediapipe_min_tracking_confidence,
            )
        except Exception as solutions_exc:
            try:
                mp, mp_tasks, vision = import_mediapipe_tasks_pose_api()
                model_asset_path = resolve_mediapipe_model_asset(args)
                if model_asset_path is None:
                    raise FileNotFoundError(
                        "MediaPipe Tasks PoseLandmarker requires a local pose_landmarker .task model. "
                        "Pass --mediapipe-model-asset or set MEDIAPIPE_POSE_MODEL."
                    )
                self.backend = "tasks"
                self.mp = mp
                options = vision.PoseLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=model_asset_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=args.mediapipe_min_detection_confidence,
                    min_pose_presence_confidence=args.mediapipe_min_detection_confidence,
                    min_tracking_confidence=args.mediapipe_min_tracking_confidence,
                    output_segmentation_masks=False,
                )
                self.pose = vision.PoseLandmarker.create_from_options(options)
            except Exception as tasks_exc:
                raise RuntimeError(
                    "Failed to initialize MediaPipe pose source.\n"
                    f"Legacy solutions error: {solutions_exc}\n"
                    f"Tasks PoseLandmarker error: {tasks_exc}\n"
                    f"{mediapipe_diagnostics()}"
                ) from tasks_exc
        self.last_joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
        self.last_joint_xy = np.zeros((self.num_joints, 2), dtype=np.float32)

    def close(self):
        self.pose.close()

    def process(self, frame_rgb):
        if self.backend == "solutions":
            results = self.pose.process(frame_rgb)
            pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        else:
            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms
            image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            results = self.pose.detect_for_video(image, timestamp_ms)
            pose_landmarks = results.pose_landmarks[0] if results.pose_landmarks else None

        if not pose_landmarks:
            return {
                "joints_3d": self.last_joints_3d.copy(),
                "joint_xy": self.last_joint_xy.copy(),
                "detected": False,
            }

        landmarks = np.asarray(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    getattr(landmark, "visibility", getattr(landmark, "presence", 1.0)),
                ]
                for landmark in pose_landmarks
            ],
            dtype=np.float32,
        )
        joints_3d, joint_xy = self.landmarks_to_joints(landmarks, self.min_visibility)
        self.last_joints_3d = joints_3d
        self.last_joint_xy = joint_xy
        return {
            "joints_3d": joints_3d,
            "joint_xy": joint_xy,
            "detected": True,
        }


def mediapipe_landmarks_to_ntu25(landmarks, min_visibility):
    def point(index):
        if landmarks[index, 3] < min_visibility:
            return None
        x = landmarks[index, 0] * 2.0 - 1.0
        y = landmarks[index, 1] * 2.0 - 1.0
        z = landmarks[index, 2]
        if not np.isfinite([x, y, z]).all():
            return None
        return np.asarray([x, y, z], dtype=np.float32)

    def average(*indices):
        values = [point(index) for index in indices]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)

    def midpoint(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return ((a + b) * 0.5).astype(np.float32)

    left_shoulder = point(11)
    right_shoulder = point(12)
    left_hip = point(23)
    right_hip = point(24)
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    spine_mid = midpoint(shoulder_center, hip_center)

    ntu_points = [
        hip_center,  # 1 spine base
        spine_mid,  # 2 spine mid
        shoulder_center,  # 3 neck
        average(0, 7, 8),  # 4 head
        left_shoulder,
        point(13),
        point(15),
        point(19),
        right_shoulder,
        point(14),
        point(16),
        point(20),
        left_hip,
        point(25),
        point(27),
        point(31),
        right_hip,
        point(26),
        point(28),
        point(32),
        shoulder_center,  # 21 spine shoulder
        point(19),
        point(21),
        point(20),
        point(22),
    ]

    joints_3d = np.zeros((25, 3), dtype=np.float32)
    for index, value in enumerate(ntu_points):
        if value is not None:
            joints_3d[index] = value
    joint_xy = np.clip(joints_3d[:, :2], -1.0, 1.0).astype(np.float32)
    return joints_3d, joint_xy


def mediapipe_landmarks_to_coco17(landmarks, min_visibility):
    """COCO-17 is a subset of MediaPipe's landmarks, so this is a plain gather.

    Nothing is synthesised here, unlike the NTU-25 mapping which has to invent
    spine and neck joints that MediaPipe never predicts.
    """

    joints_3d = np.zeros((17, 3), dtype=np.float32)
    for index, landmark_index in enumerate(COCO17_FROM_MEDIAPIPE):
        if landmarks[landmark_index, 3] < min_visibility:
            continue
        point = np.asarray(
            [
                landmarks[landmark_index, 0] * 2.0 - 1.0,
                landmarks[landmark_index, 1] * 2.0 - 1.0,
                landmarks[landmark_index, 2],
            ],
            dtype=np.float32,
        )
        if not np.isfinite(point).all():
            continue
        joints_3d[index] = point

    joint_xy = np.clip(joints_3d[:, :2], -1.0, 1.0).astype(np.float32)
    return joints_3d, joint_xy


def build_rtmpose_skeleton_inputs(clip, device):
    """[3,T,17,2] clip -> CTR-GCN skeleton batch and the fusion head's joint_xy.

    joint_xy takes person 0 only: it places the pose maps on the RS grid, which
    holds one body. This mirrors tools/build_coco17_pose_features.py.
    """

    skeleton = torch.from_numpy(clip).unsqueeze(0).to(device=device, dtype=torch.float32)
    joint_xy = np.ascontiguousarray(clip[:2, :, :, 0].transpose(1, 2, 0))
    joint_xy = torch.from_numpy(joint_xy).unsqueeze(0).to(device=device, dtype=torch.float32)
    return skeleton, joint_xy


def resize_skeleton_time(skeletons, window_size):
    """Stretch the captured window to the length the backbone was trained on."""

    batch, channels, frames, joints, persons = skeletons.shape
    x = skeletons.permute(0, 1, 3, 4, 2).reshape(batch, channels * joints * persons, frames)
    x = torch.nn.functional.interpolate(x, size=window_size, mode="linear", align_corners=False)
    x = x.reshape(batch, channels, joints, persons, window_size)
    return x.permute(0, 1, 4, 2, 3).contiguous()


def resize_pose_feature_time(pose, frames):
    """Resample hooked pose features back to the steps the fusion head wants."""

    batch, persons, channels, steps, joints = pose.shape
    x = pose.permute(0, 1, 2, 4, 3).reshape(-1, 1, steps)
    x = torch.nn.functional.interpolate(x, size=frames, mode="linear", align_corners=False)
    x = x.reshape(batch, persons, channels, joints, frames)
    return x.permute(0, 1, 2, 4, 3).contiguous()


def ctrgcn_pose_features(model, captured, skeletons, args):
    """Pose features [B, M, C, T, V] for the fusion head.

    The COCO-17 backbone was trained on 64-frame windows, so the captured steps
    are stretched on the way in and the hooked features resampled back on the
    way out - the same two resizes the offline feature builder performs.
    """

    frames = skeletons.shape[2]
    window_size = getattr(args, "ctrgcn_window_size", None)
    if window_size and window_size != frames:
        skeletons = resize_skeleton_time(skeletons, window_size)

    pose = ctrgcn_pose_from_model(model, captured, skeletons, args)
    if pose.shape[3] != frames:
        pose = resize_pose_feature_time(pose, frames)
    return pose


def build_zero_pose_inputs(batch_size, frames, device, num_joints=25):
    # Laptop webcam RGB has no Kinect/ETRI skeleton stream.
    pose = torch.zeros(batch_size, 2, 64, frames, num_joints, dtype=torch.float32, device=device)
    joint_xy = torch.zeros(batch_size, frames, num_joints, 2, dtype=torch.float32, device=device)
    return pose, joint_xy


def build_mediapipe_skeleton_inputs(skeleton_buffer, device):
    joints_3d = np.stack([sample["joints_3d"] for sample in skeleton_buffer], axis=0).astype(np.float32)
    joint_xy = np.stack([sample["joint_xy"] for sample in skeleton_buffer], axis=0).astype(np.float32)

    skeleton = np.zeros((3, joints_3d.shape[0], joints_3d.shape[1], 2), dtype=np.float32)
    skeleton[:, :, :, 0] = joints_3d.transpose(2, 0, 1)
    detected_frames = sum(1 for sample in skeleton_buffer if sample["detected"])

    return (
        torch.from_numpy(skeleton).unsqueeze(0).to(device=device, dtype=torch.float32),
        torch.from_numpy(joint_xy).unsqueeze(0).to(device=device, dtype=torch.float32),
        detected_frames,
    )


def action_duration_type(label, args):
    return getattr(args, "action_duration_types", DEFAULT_ACTION_DURATION_TYPES).get(int(label), "S")


def is_long_action(label, args):
    return action_duration_type(label, args).upper() == "L"


def history_duration_seconds(history_buffer):
    if len(history_buffer) < 2:
        return 0.0
    return max(0.0, float(history_buffer[-1]["timestamp"] - history_buffer[0]["timestamp"]))


def trim_history_buffer(history_buffer, args, now):
    if args.temporal_strategy == "uniform3s":
        max_window = args.uniform_window_seconds
    else:
        max_window = max(args.short_window_seconds, args.long_window_seconds)
    keep_seconds = max_window + 1.0
    while len(history_buffer) > args.frames and now - history_buffer[0]["timestamp"] > keep_seconds:
        history_buffer.popleft()


def history_window_ready(history_buffer, window_seconds, args):
    return len(history_buffer) >= args.frames and history_duration_seconds(history_buffer) >= window_seconds


def active_window_seconds(args):
    if args.temporal_strategy == "uniform3s":
        return args.uniform_window_seconds
    return args.short_window_seconds


def warmup_frame_count(history_buffer, args):
    if not history_buffer:
        return 0
    progress = min(1.0, history_duration_seconds(history_buffer) / active_window_seconds(args))
    return min(args.frames, max(1, int(round(progress * args.frames))))


def sample_history_window(history_buffer, window_seconds, frame_count):
    samples = list(history_buffer)
    if not samples:
        raise ValueError("Cannot sample an empty realtime history.")

    end_time = samples[-1]["timestamp"]
    start_time = end_time - window_seconds
    timestamps = [sample["timestamp"] for sample in samples]
    targets = np.linspace(start_time, end_time, num=frame_count)

    selected = []
    cursor = 0
    last_index = len(samples) - 1
    for target in targets:
        while cursor < last_index and timestamps[cursor] < target:
            cursor += 1
        if cursor > 0:
            previous_index = cursor - 1
            if abs(timestamps[previous_index] - target) <= abs(timestamps[cursor] - target):
                selected.append(samples[previous_index])
                continue
        selected.append(samples[cursor])

    frames = [sample["frame_rgb"] for sample in selected]
    skeletons = [sample["skeleton"] for sample in selected if sample.get("skeleton") is not None]
    return frames, skeletons, selected


def latest_history_ready(history_buffer, args):
    if len(history_buffer) < args.frames:
        return False
    if args.runtime_pose_source != "mediapipe":
        return True
    return all(sample.get("skeleton") is not None for sample in list(history_buffer)[-args.frames :])


def latest_history_window(history_buffer, frame_count):
    samples = list(history_buffer)[-frame_count:]
    frames = [sample["frame_rgb"] for sample in samples]
    skeletons = [sample["skeleton"] for sample in samples if sample.get("skeleton") is not None]
    return frames, skeletons, samples


def finish_end_to_end_timer(start, device):
    """Finish timing after all asynchronous accelerator work is complete."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def score_prediction_window(
    frames,
    skeleton_samples,
    args,
    history_samples,
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
):
    x3d_clip = x3d_tensor_from_frames(
        frames,
        int(x3d_cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE),
        x3d_cfg.TRANSFORM.MEAN,
        x3d_cfg.TRANSFORM.STD,
    ).unsqueeze(0).to(device, non_blocking=True)
    yolo_sample_index = len(frames) // 2
    yolo_frame = frames[yolo_sample_index]
    yolo_history_sample = (
        history_samples[yolo_sample_index]
        if history_samples and yolo_sample_index < len(history_samples)
        else None
    )
    detected_pose_frames = 0
    gate_joints = None
    if args.runtime_pose_source == "mediapipe":
        skeletons, joint_xy, detected_pose_frames = build_mediapipe_skeleton_inputs(skeleton_samples, device)
        gate_joints = [sample["joints_3d"] for sample in skeleton_samples] or None
    elif args.runtime_pose_source != "rtmpose":
        pose_features, joint_xy = build_zero_pose_inputs(
            batch_size=1,
            frames=args.frames,
            device=device,
            num_joints=getattr(args, "num_joints", 25),
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    if args.runtime_pose_source == "rtmpose":
        # Realtime history entries already carry RTMPose results. The fallback
        # inside skeleton_from_frames only serves callers that supply uncached
        # samples, such as isolated tests and offline utilities.
        clip, detected_pose_frames = args.pose_estimator.skeleton_from_frames(frames, history_samples)
        skeletons, joint_xy = build_rtmpose_skeleton_inputs(clip, device)
        gate_joints = list(np.ascontiguousarray(clip[:2, :, :, 0].transpose(1, 2, 0)))
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        cached_object_map = (
            yolo_history_sample.get("object_map")
            if yolo_history_sample is not None
            else None
        )
        if cached_object_map is None:
            # Offline/isolated callers do not populate realtime history caches.
            object_maps = object_map_runner([yolo_frame])
            watched_detections = [
                dict(item) for item in object_map_runner.last_watched_detections
            ]
        else:
            object_maps = cached_object_map.unsqueeze(0).to(
                device, non_blocking=True
            )
            watched_detections = [
                dict(item)
                for item in yolo_history_sample.get(
                    "yolo_watched_detections", []
                )
            ]
        video_features = x3d_features_from_model(x3d_model, x3d_captured, x3d_clip, args)
        if args.runtime_pose_source in ("mediapipe", "rtmpose"):
            pose_features = ctrgcn_pose_features(ctrgcn_model, ctrgcn_captured, skeletons, args)
        logits = clipgcn_model(video_features, pose_features, object_maps, joint_xy)
        # The decision gate computes H from plain cosine. The [0,1] cosine is
        # retained only for diagnostics; unseen/class calibration is ranking-only
        # and must not leak into the entropy measurement.
        raw_cosine_scores = logits_to_cosine(clipgcn_model, logits.float())
        cosine_scores = logits_to_unit_cosine_scores(clipgcn_model, logits)
        prediction_scores, used_scale = apply_unseen_score_scale(
            clipgcn_model,
            logits,
            candidate_labels,
            unseen_labels=unseen_labels,
            unseen_score_scale=args.unseen_score_scale,
        )
        prediction_scores, cosine_scores, used_class_adjustments = apply_class_score_adjustments(
            prediction_scores,
            cosine_scores,
            candidate_labels,
            args,
            used_unseen_scale=used_scale,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "prediction_scores": prediction_scores[0].detach().cpu(),
        "cosine_scores": cosine_scores[0].detach().cpu(),
        "raw_cosine_scores": raw_cosine_scores[0].detach().cpu(),
        "object_map": object_maps.detach().float().cpu(),
        "joints": gate_joints,
        "elapsed": elapsed,
        "used_scale": used_scale,
        "used_class_score_adjustments": used_class_adjustments,
        "detected_pose_frames": detected_pose_frames,
        "yolo_watched_detections": watched_detections,
    }


def apply_class_score_adjustments(prediction_scores, cosine_scores, candidate_labels, args, used_unseen_scale=False):
    adjustments = getattr(args, "class_score_adjustments", {}) or {}
    active = {
        int(label): value
        for label, value in adjustments.items()
        if int(label) in {int(candidate_label) for candidate_label in candidate_labels}
    }
    if not active:
        return prediction_scores, cosine_scores, False

    adjusted_cosine = cosine_scores.clone()
    adjusted_prediction = prediction_scores.clone() if used_unseen_scale else cosine_scores.clone()
    for index, label in enumerate(candidate_labels):
        item = active.get(int(label))
        if not item:
            continue
        scale = float(item.get("scale", 1.0))
        bias = float(item.get("bias", 0.0))
        adjusted_cosine[:, index] = (adjusted_cosine[:, index] * scale + bias).clamp(0.0, 1.0)
        adjusted_prediction[:, index] = adjusted_prediction[:, index] * scale + bias
        if not used_unseen_scale:
            adjusted_prediction[:, index] = adjusted_cosine[:, index]

    return adjusted_prediction, adjusted_cosine, True


def build_prediction_from_scores(score_result, args, candidate_labels, temporal_mode):
    top_k = min(args.top_k, len(candidate_labels))
    top_scores, top_indices = torch.topk(score_result["prediction_scores"], k=top_k)
    labels = [int(candidate_labels[index]) for index in top_indices.detach().cpu().tolist()]
    scores = [float(score_result["cosine_scores"][index].detach().cpu()) for index in top_indices.tolist()]
    ranking_scores = [float(score) for score in top_scores.detach().cpu().tolist()]
    gate_info = evaluate_decision_condition(
        score_result,
        args,
        candidate_labels,
        top_label=labels[0],
    )
    gate_label = int(gate_info["label"])
    display_labels = labels
    display_scores = scores
    display_ranking_scores = ranking_scores
    if gate_label in (UNKNOWN_LABEL, ANALYSING_LABEL):
        keep_count = max(0, top_k - 1)
        display_labels = [gate_label] + labels[:keep_count]
        display_scores = [gate_info["entropy"]] + scores[:keep_count]
        display_ranking_scores = [gate_info["entropy"]] + ranking_scores[:keep_count]
    elif labels and gate_label != labels[0]:
        # Keep this defensive path for any future policy that accepts a
        # candidate other than the calibrated top-1.
        position = labels.index(gate_label) if gate_label in labels else None
        order = [position] if position is not None else []
        order += [index for index in range(len(labels)) if index != position]
        display_labels = [labels[index] for index in order][:top_k]
        display_scores = [scores[index] for index in order][:top_k]
        display_ranking_scores = [ranking_scores[index] for index in order][:top_k]
    raw_labels = labels[:top_k]
    raw_scores = scores[:top_k]
    raw_ranking_scores = ranking_scores[:top_k]
    return {
        "labels": display_labels,
        "scores": display_scores,
        "ranking_scores": display_ranking_scores,
        "raw_labels": raw_labels,
        "raw_scores": raw_scores,
        "raw_ranking_scores": raw_ranking_scores,
        "unknown": gate_label == UNKNOWN_LABEL,
        "analysing": gate_label == ANALYSING_LABEL,
        "unknown_reason": gate_info["reason"] if gate_label == UNKNOWN_LABEL else None,
        "analysing_reason": gate_info["reason"] if gate_label == ANALYSING_LABEL else None,
        "unknown_entropy": gate_info["entropy"],
        "unknown_top1_score": gate_info["cosine"],
        "gate": gate_info,
        "elapsed": score_result["elapsed"],
        "used_scale": score_result["used_scale"],
        "used_class_score_adjustments": score_result.get("used_class_score_adjustments", False),
        "detected_pose_frames": score_result["detected_pose_frames"],
        "yolo_watched_detections": [dict(item) for item in score_result.get("yolo_watched_detections", [])],
        "temporal_mode": temporal_mode,
    }


def evaluate_decision_condition(score_result, args, candidate_labels, top_label):
    """Run the single configured H-then-prototype decision policy."""

    gate = getattr(args, "decision_gate", None)
    raw_cosine_scores = score_result.get(
        "raw_cosine_scores", score_result["cosine_scores"] * 2.0 - 1.0
    ).float()
    unit_cosine_scores = ((raw_cosine_scores + 1.0) * 0.5).clamp(0.0, 1.0)
    if gate is None:
        gate = PersonEntropyDecisionGate(
            candidate_labels,
            getattr(args, "unseen_label_set", set()),
            {"enabled": False},
        )
    return gate(
        raw_cosine_scores,
        unit_cosine_scores,
        top_label,
        person_present=getattr(args, "current_person_present", None),
        object_map=score_result.get("object_map"),
    )


def merge_short_long_scores(short_result, long_result, args, candidate_labels):
    prediction_scores = short_result["prediction_scores"].clone()
    cosine_scores = short_result["cosine_scores"].clone()
    raw_cosine_scores = short_result["raw_cosine_scores"].clone()
    for index, label in enumerate(candidate_labels):
        if is_long_action(label, args):
            prediction_scores[index] = long_result["prediction_scores"][index]
            cosine_scores[index] = long_result["cosine_scores"][index]
            raw_cosine_scores[index] = long_result["raw_cosine_scores"][index]
    watched_by_name = {}
    for result in (short_result, long_result):
        for detection in result.get("yolo_watched_detections", []):
            name = str(detection["name"])
            confidence = float(detection["confidence"])
            watched_by_name[name] = max(watched_by_name.get(name, 0.0), confidence)
    watched_detections = [
        {"name": name, "confidence": watched_by_name[name]}
        for name in ("stove", "biscuits", "pot")
        if name in watched_by_name
    ]
    return {
        "prediction_scores": prediction_scores,
        "cosine_scores": cosine_scores,
        "raw_cosine_scores": raw_cosine_scores,
        # Motion and person presence come from the longer window, which is the
        # one that actually shows whether the person moved.
        "object_map": long_result.get("object_map"),
        "joints": long_result.get("joints"),
        "elapsed": short_result["elapsed"] + long_result["elapsed"],
        "used_scale": short_result["used_scale"],
        "detected_pose_frames": long_result["detected_pose_frames"],
        "yolo_watched_detections": watched_detections,
    }


def run_prediction(
    frame_buffer,
    skeleton_buffer,
    args,
    history_samples,
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
):
    score_result = score_prediction_window(
        list(frame_buffer),
        list(skeleton_buffer),
        args,
        history_samples,
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
    return build_prediction_from_scores(score_result, args, candidate_labels, temporal_mode="13f")


def run_last13_history_prediction(
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
):
    frames, skeletons, samples = latest_history_window(history_buffer, args.frames)
    return run_prediction(
        frames,
        skeletons,
        args,
        samples,
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


def run_temporal_prediction(
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
):
    if args.temporal_strategy == "uniform3s":
        frames, skeletons, samples = sample_history_window(
            history_buffer,
            args.uniform_window_seconds,
            args.frames,
        )
        result = score_prediction_window(
            frames,
            skeletons,
            args,
            samples,
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
        return build_prediction_from_scores(result, args, candidate_labels, temporal_mode="3s")

    short_frames, short_skeletons, short_samples = sample_history_window(
        history_buffer,
        args.short_window_seconds,
        args.frames,
    )
    short_result = score_prediction_window(
        short_frames,
        short_skeletons,
        args,
        short_samples,
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
    # Only peek at the ranking here; the final short/long choice must pass
    # through the decision policy exactly once.
    rerank_count = min(args.long_rerank_top_k, len(candidate_labels))
    _top_scores, top_indices = torch.topk(short_result["prediction_scores"], k=rerank_count)
    probe_labels = [int(candidate_labels[index]) for index in top_indices.tolist()]
    needs_long_window = (
        args.temporal_strategy == "short-long"
        and any(is_long_action(label, args) for label in probe_labels)
        and history_window_ready(history_buffer, args.long_window_seconds, args)
    )
    if not needs_long_window:
        return build_prediction_from_scores(short_result, args, candidate_labels, temporal_mode="2s")

    long_frames, long_skeletons, long_samples = sample_history_window(
        history_buffer,
        args.long_window_seconds,
        args.frames,
    )
    long_result = score_prediction_window(
        long_frames,
        long_skeletons,
        args,
        long_samples,
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
    merged_result = merge_short_long_scores(short_result, long_result, args, candidate_labels)
    return build_prediction_from_scores(merged_result, args, candidate_labels, temporal_mode="2s+4s")


def truncate_text(value, max_chars=46):
    value = str(value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def prediction_label_text(label, args, max_chars=46):
    label = int(label)
    if label == UNKNOWN_LABEL:
        return "UNKNOWN"
    if label == ANALYSING_LABEL:
        return "ANALYSING"
    name = getattr(args, "label_display_names", {}).get(label)
    suffix = " (unseen)" if is_unseen_label(label, args) else ""
    if not name:
        return f"class {label}{suffix}"
    return f"{label} {truncate_text(name, max_chars=max_chars)}{suffix}"


def is_unseen_label(label, args):
    if int(label) in (UNKNOWN_LABEL, ANALYSING_LABEL):
        return False
    return int(label) in getattr(args, "unseen_label_set", set())


def prediction_color(label, rank, args):
    if int(label) == UNKNOWN_LABEL:
        return (80, 180, 255)
    if int(label) == ANALYSING_LABEL:
        return (0, 215, 255)
    if is_unseen_label(label, args):
        return (70, 210, 255) if rank == 1 else (120, 230, 255)
    return (80, 255, 120) if rank == 1 else (230, 230, 230)


def prediction_score_text(label, score, prediction):
    if int(label) in (UNKNOWN_LABEL, ANALYSING_LABEL):
        gate = prediction.get("gate") or {}
        entropy = float(prediction.get("unknown_entropy", score))
        cosine = float(prediction.get("unknown_top1_score", 0.0))
        fallback = "unknown" if int(label) == UNKNOWN_LABEL else "analysing"
        return f"{gate.get('reason', fallback)} H={entropy:.2f} proto={cosine:.2f}"
    return f"{score:.4f}"


def gate_status_text(prediction):
    gate = prediction.get("gate") or {}
    if not gate:
        return ""
    split_name = "unseen" if gate.get("is_unseen", False) else "seen"
    return (
        f"gate={gate.get('decision', '-')} ({gate.get('reason', '-')}) "
        f"person={int(bool(gate.get('person', False)))} "
        f"H={float(gate.get('entropy', 0.0)):.2f} "
        f"proto={float(gate.get('cosine', 0.0)):.2f} split={split_name}"
    )


def vote_ranked_prediction(prediction_history, args):
    if not prediction_history:
        return None

    latest = prediction_history[-1]
    window = max(1, int(getattr(args, "display_filter_window", 1)))
    recent = list(prediction_history)[-window:]
    latest_labels = [int(label) for label in latest.get("labels", [])]
    if latest_labels and latest_labels[0] in (ANALYSING_LABEL, UNKNOWN_LABEL):
        # Gate state has higher priority than label smoothing. In particular,
        # no-person and high-H decisions must become ANALYSING immediately,
        # rather than waiting for a window of old class votes to expire.
        return latest
    recent = [
        item
        for item in recent
        if not item.get("labels")
        or int(item["labels"][0]) not in (ANALYSING_LABEL, UNKNOWN_LABEL)
    ]
    if window <= 1 or len(recent) <= 1:
        return latest

    max_rows = min(
        int(getattr(args, "top_k", len(latest.get("labels", [])))),
        max(len(item.get("labels", [])) for item in recent),
    )
    voted_labels = []
    voted_scores = []
    voted_ranking_scores = []
    used_labels = set()

    for rank_index in range(max_rows):
        stats = {}
        for history_index, item in enumerate(recent):
            labels = item.get("labels", [])
            if rank_index >= len(labels):
                continue
            label = int(labels[rank_index])
            score = float(item.get("scores", [0.0] * len(labels))[rank_index])
            ranking_scores = item.get("ranking_scores", item.get("scores", []))
            ranking_score = float(ranking_scores[rank_index]) if rank_index < len(ranking_scores) else score
            entry = stats.setdefault(
                label,
                {
                    "count": 0,
                    "score_sum": 0.0,
                    "ranking_score_sum": 0.0,
                    "latest_seen": -1,
                },
            )
            entry["count"] += 1
            entry["score_sum"] += score
            entry["ranking_score_sum"] += ranking_score
            entry["latest_seen"] = history_index

        if not stats:
            continue

        ranked = sorted(
            stats.items(),
            key=lambda item: (
                item[1]["count"],
                item[1]["ranking_score_sum"] / max(1, item[1]["count"]),
                item[1]["latest_seen"],
            ),
            reverse=True,
        )
        selected_label, selected_stats = next(
            ((label, data) for label, data in ranked if label not in used_labels),
            ranked[0],
        )
        used_labels.add(selected_label)
        count = max(1, selected_stats["count"])
        voted_labels.append(int(selected_label))
        voted_scores.append(float(selected_stats["score_sum"] / count))
        voted_ranking_scores.append(float(selected_stats["ranking_score_sum"] / count))

    filtered = dict(latest)
    filtered["labels"] = voted_labels
    filtered["scores"] = voted_scores
    filtered["ranking_scores"] = voted_ranking_scores
    if voted_labels:
        # The voted state can differ from the newest raw state. Keep H,
        # prototype and reason from a window that actually produced the voted
        # top row so the overlay never explains UNKNOWN as "accepted" (or the
        # reverse).
        for item in reversed(recent):
            item_labels = [int(label) for label in item.get("labels", [])]
            if item_labels and item_labels[0] == voted_labels[0]:
                filtered["gate"] = dict(item.get("gate") or {})
                filtered["unknown_entropy"] = item.get("unknown_entropy", 0.0)
                filtered["unknown_top1_score"] = item.get("unknown_top1_score", 0.0)
                break
    filtered["unknown"] = UNKNOWN_LABEL in voted_labels
    filtered["analysing"] = ANALYSING_LABEL in voted_labels
    for sentinel_label, reason_key, fallback_reason in (
        (UNKNOWN_LABEL, "unknown_reason", "unknown"),
        (ANALYSING_LABEL, "analysing_reason", "analysing"),
    ):
        if sentinel_label not in voted_labels:
            filtered[reason_key] = None
            continue
        reason_counts = {}
        for item in recent:
            if sentinel_label not in [int(label) for label in item.get("labels", [])]:
                continue
            reason = item.get(reason_key) or fallback_reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason_counts:
            filtered[reason_key] = max(reason_counts.items(), key=lambda item: item[1])[0]
    filtered["display_filter_window"] = window
    filtered["display_filter_count"] = len(recent)
    filtered["raw_temporal_mode"] = latest.get("temporal_mode", "13f")
    filtered["temporal_mode"] = f"{latest.get('temporal_mode', '13f')} vote{len(recent)}/{window}"
    return filtered


def _box_iou(first, second):
    intersection = (
        max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
        * max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def draw_person_boxes(
    frame_bgr,
    person_detections,
    tracked_person,
    source_frame_shape,
):
    """Draw every YOLO person; the target selected for PID is red."""

    if source_frame_shape is None:
        source_height, source_width = frame_bgr.shape[:2]
    else:
        source_height, source_width = source_frame_shape[:2]
    scale_x = frame_bgr.shape[1] / max(1.0, float(source_width))
    scale_y = frame_bgr.shape[0] / max(1.0, float(source_height))
    tracked_bbox = tracked_person[:4] if tracked_person is not None else None

    for detection in person_detections or []:
        try:
            bbox = (
                float(detection["x1"]),
                float(detection["y1"]),
                float(detection["x2"]),
                float(detection["y2"]),
            )
            confidence = float(detection["confidence"])
        except (KeyError, TypeError, ValueError):
            continue

        # The tracked target is drawn once, in red, after all other people.
        if tracked_bbox is not None and _box_iou(bbox, tracked_bbox) >= 0.95:
            continue
        x1, y1, x2, y2 = (
            int(round(bbox[0] * scale_x)),
            int(round(bbox[1] * scale_y)),
            int(round(bbox[2] * scale_x)),
            int(round(bbox[3] * scale_y)),
        )
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (80, 255, 120), 2)
        cv2.putText(
            frame_bgr,
            f"person {confidence:.2f}",
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )

    if tracked_bbox is not None:
        x1, y1, x2, y2 = (
            int(round(tracked_bbox[0] * scale_x)),
            int(round(tracked_bbox[1] * scale_y)),
            int(round(tracked_bbox[2] * scale_x)),
            int(round(tracked_bbox[3] * scale_y)),
        )
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        confidence = float(tracked_person[4])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(frame_bgr, center, 6, (0, 0, 255), thickness=-1)
        cv2.putText(
            frame_bgr,
            f"TRACKING {confidence:.2f}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # A short marker shows the visual-control setpoint without obscuring
        # the whole image.
        image_center_x = frame_bgr.shape[1] // 2
        cv2.line(
            frame_bgr,
            (image_center_x, frame_bgr.shape[0] - 28),
            (image_center_x, frame_bgr.shape[0]),
            (255, 220, 80),
            2,
        )
    return frame_bgr


def draw_overlay(
    frame_bgr,
    prediction,
    frame_count,
    args,
    person_detections=None,
    tracked_person=None,
    source_frame_shape=None,
):
    overlay = frame_bgr.copy()
    shown_rows = 1 if prediction is None else len(prediction.get("labels", []))
    watched_detections = [] if prediction is None else prediction.get("yolo_watched_detections", [])
    detected_names = {str(item["name"]) for item in watched_detections}
    watched_flags_text = "  ".join(
        f"{name}={int(name in detected_names)}" for name in ("stove", "biscuits", "pot")
    )
    overlay_height = max(175, 135 + shown_rows * 25)
    overlay_width = min(frame_bgr.shape[1], 920)
    cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    cv2.putText(
        frame_bgr,
        "CLIPGCN realtime",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if args.temporal_strategy == "last13":
        window_text = f"latest_contiguous={args.frames}"
    elif args.temporal_strategy == "uniform3s":
        window_text = f"window={args.uniform_window_seconds:g}s"
    else:
        window_text = f"window={args.short_window_seconds:g}s/{args.long_window_seconds:g}s"
    cv2.putText(
        frame_bgr,
        f"frames={args.frames}  {window_text}  pose={args.runtime_pose_source}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )

    if prediction is None:
        cv2.putText(
            frame_bgr,
            f"YOLO objects: {watched_flags_text}",
            (18, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )
        text = f"warming up: {frame_count}/{args.frames}"
        cv2.putText(frame_bgr, text, (18, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2, cv2.LINE_AA)
        if not args.hide_person_boxes:
            draw_person_boxes(
                frame_bgr,
                person_detections,
                tracked_person,
                source_frame_shape,
            )
        return frame_bgr

    core_latency_ms = prediction["elapsed"] * 1000.0
    end_to_end_latency_ms = prediction.get("end_to_end_elapsed", prediction["elapsed"]) * 1000.0
    cv2.putText(
        frame_bgr,
        (
            f"E2E {end_to_end_latency_ms:.1f} ms  core {core_latency_ms:.1f} ms  "
            f"pose_frames {prediction['detected_pose_frames']}/{args.frames}  "
            f"{gate_status_text(prediction)}  mode={prediction.get('temporal_mode', '13f')}"
        ),
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"YOLO objects: {watched_flags_text}",
        (18, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )
    ranking_base_y = 117
    for rank, (label, score) in enumerate(zip(prediction["labels"], prediction["scores"]), start=1):
        y = ranking_base_y + rank * 24
        color = prediction_color(label, rank, args)
        cv2.putText(
            frame_bgr,
            f"{rank}. {prediction_label_text(label, args, max_chars=62)}: {prediction_score_text(label, score, prediction)}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
    if not args.hide_person_boxes:
        draw_person_boxes(
            frame_bgr,
            person_detections,
            tracked_person,
            source_frame_shape,
        )
    return frame_bgr


def resize_for_display(frame_bgr, args):
    """Resize only the rendered frame; model inputs keep their native resolution."""

    source_height, source_width = frame_bgr.shape[:2]
    display_width = int(args.display_width)
    display_height = int(args.display_height)
    if display_width == 0 and display_height == 0:
        return frame_bgr
    if display_width == 0:
        display_width = max(1, round(source_width * display_height / source_height))
    elif display_height == 0:
        display_height = max(1, round(source_height * display_width / source_width))
    if display_width == source_width and display_height == source_height:
        return frame_bgr
    return cv2.resize(frame_bgr, (display_width, display_height), interpolation=cv2.INTER_LINEAR)


def print_prediction(prediction, args):
    if prediction is None:
        return
    parts = [
        f"{rank}. {prediction_label_text(label, args)}={prediction_score_text(label, score, prediction)}"
        for rank, (label, score) in enumerate(zip(prediction["labels"], prediction["scores"]), start=1)
    ]
    mode = prediction.get("temporal_mode", "13f")
    core_latency_ms = prediction["elapsed"] * 1000.0
    end_to_end_latency_ms = prediction.get("end_to_end_elapsed", prediction["elapsed"]) * 1000.0
    watched_detections = prediction.get("yolo_watched_detections", [])
    detected_names = {str(item["name"]) for item in watched_detections}
    watched_text = " | YOLO " + ", ".join(
        f"{name}={int(name in detected_names)}" for name in ("stove", "biscuits", "pot")
    )
    gate_text = gate_status_text(prediction)
    print(
        f"[E2E {end_to_end_latency_ms:.1f} ms | core {core_latency_ms:.1f} ms | {mode}] "
        + (f"{gate_text} | " if gate_text else "")
        + " | ".join(parts)
        + watched_text,
        flush=True,
    )


def main(args=None, capture_factory=None):
    args = parse_args() if args is None else args
    validate_args(args)
    args.action_duration_types = DEFAULT_ACTION_DURATION_TYPES
    args = maybe_enable_headless(args)

    # The config decides the skeleton layout, and the layout decides which
    # estimator is the right one, so it has to be read first.
    config_path = resolve_existing_path(args.config, label="config")
    config = load_config(config_path)
    args = resolve_pose_backend(args, config)
    args.runtime_pose_source = resolve_runtime_pose_source(args)
    apply_runtime_device_override(args, config)
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
    try:
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
    except FileNotFoundError as exc:
        if args.pose_layout != "coco17":
            raise
        raise FileNotFoundError(
            f"{exc}\n"
            f"This checkpoint uses the COCO-17 skeleton, so it needs the RTMPose CTR-GCN backbone "
            f"({CTRGCN_17_ROOT}) rather than the 25-joint ETRI one. Copy over "
            "config/etri-coco17/ctrgcn_joint_coco17_13.yaml and "
            "work_dir/etri_coco17/ctrgcn_joint_coco17_13/runs-64-768.pt, or point "
            "--ctrgcn-root/--ctrgcn-config/--ctrgcn-weights at wherever they live."
        ) from exc
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
    # Both estimators run once per captured frame. RTMPose is held on args so
    # prediction-window helpers can pack the keypoints cached on history items.
    pose_source = MediaPipePoseSource(args) if args.runtime_pose_source == "mediapipe" else None
    args.pose_estimator = RTMPoseSource(args) if args.runtime_pose_source == "rtmpose" else None

    person_publisher_class = None
    if getattr(args, "enable_person_follow_output", False):
        if yolo_model is None:
            raise RuntimeError(
                "--enable-person-follow-output requires YOLO; remove --no-yolo "
                "and provide valid YOLO weights."
            )
        try:
            from person_follow_demo.har_person_detection_publisher import (
                HarPersonDetectionPublisher,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Person-follow output needs /workspace/person_follow_ws. Build it "
                "with colcon and source install/setup.bash before ros_realtime.py."
            ) from exc
        person_publisher_class = HarPersonDetectionPublisher

    cap = (capture_factory or open_camera)(args)
    person_detection_publisher = None
    if person_publisher_class is not None:
        ros_node = getattr(cap, "node", None)
        if ros_node is None:
            cap.release()
            raise RuntimeError(
                "--enable-person-follow-output is only supported by a capture "
                "source exposing an rclpy node, such as ros_realtime.py."
            )
        person_detection_publisher = person_publisher_class(
            ros_node,
            topic=args.person_detection_topic,
            frame_id=args.person_detection_frame_id,
            confidence_threshold=args.person_confidence_threshold,
            lock_iou_threshold=args.person_lock_iou_threshold,
            max_center_jump_fraction=args.person_max_center_jump,
            max_lost_frames=args.person_max_lost_frames,
        )
    source_name = getattr(cap, "source_name", "webcam")
    fusion_config = (config.get("model") or {}).get("fusion") or {}
    print(f"Realtime CLIPGCN {source_name} inference")
    if getattr(args, "robot_namespace", None):
        print(f"  robot_namespace: {args.robot_namespace}")
    print(f"  checkpoint: {checkpoint_path}")
    print(
        "  fusion: "
        f"{fusion_config.get('reducer', 'attention_pool')} + "
        f"{fusion_config.get('object_fusion', 'cross_attention')} "
        f"({fusion_config.get('video_channels', 25)} video + {args.num_joints} joints)"
    )
    print(f"  text prototypes: {len(config['data']['text'].get('prompt_templates') or ['{global_description}'])} per class")
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
        print("  decision_states: class / ANALYSING")
    else:
        print("  decision_gate: disabled in YAML")
    print(f"  pose_source: {args.pose_source}")
    if args.pose_estimator is not None:
        print(f"  rtmpose: mode={args.pose_estimator.mode} device={args.pose_estimator.device}")
    print(f"  pose_layout: {args.pose_layout} ({args.num_joints} joints)")
    print(f"  ctrgcn_weights: {args.ctrgcn_weights}")
    if args.ctrgcn_window_size:
        print(f"  ctrgcn_window: {args.frames} -> {args.ctrgcn_window_size} frames in, features back to {args.frames}")
    print(f"  runtime_pose_source: {args.runtime_pose_source}")
    print(f"  temporal_strategy: {args.temporal_strategy}")
    if args.temporal_strategy == "last13":
        print(f"  latest_contiguous_frames: {args.frames}")
    elif args.temporal_strategy == "uniform3s":
        print(f"  uniform_window_seconds: {args.uniform_window_seconds:g}")
    else:
        print(f"  short_window_seconds: {args.short_window_seconds:g}")
        print(f"  long_window_seconds: {args.long_window_seconds:g}")
    source_description = getattr(cap, "source_description", None)
    if source_description:
        print(f"  input: {source_description}")
    else:
        print(f"  camera_index: {args.camera_index}")
    print("  YOLO overlay targets: stove, biscuits, pot")
    if person_detection_publisher is not None:
        print(
            "  person_follow_output: "
            f"{args.person_detection_topic} "
            f"(conf>={args.person_confidence_threshold:g})"
        )
    else:
        print("  person_follow_output: disabled")
    print("  latency: E2E=newest raw frame to voted prediction; core=model section only")
    print("  quit: press q or ESC")

    history_buffer = deque()
    prediction = None
    display_prediction = None
    prediction_history = deque(maxlen=args.display_filter_window)
    captured_frames = 0
    current_person_detections = []
    tracked_person = None
    display_window_initialized = False

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            # Input acquisition is excluded; everything from the newly read raw
            # frame through pose extraction and the final voted prediction is E2E.
            frame_processing_start = time.perf_counter()
            captured_frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = time.monotonic()

            # Run YOLO exactly once for each new camera frame. The current
            # person boxes feed the visual tracker immediately; CLIPGCN later
            # reuses the cached object map for whichever temporal sample it
            # selects, avoiding a second detector pass.
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                current_object_maps = object_map_runner([frame_rgb])
            current_person_detections = [
                dict(item) for item in object_map_runner.last_person_detections
            ]
            args.current_person_present = bool(current_person_detections)
            current_watched_detections = [
                dict(item) for item in object_map_runner.last_watched_detections
            ]
            if person_detection_publisher is not None:
                tracked_person = person_detection_publisher.publish_from_yolo(
                    current_person_detections,
                    frame_bgr.shape,
                    stamp=getattr(cap, "latest_stamp", None),
                )

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
                # RTMPose is stateless per frame, so estimating here and picking
                # the window's frames afterwards gives exactly the same
                # keypoints as estimating the picked frames - but the cost is
                # spread over the capture loop instead of landing inside one
                # prediction. Two windows that share frames also pay once.
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
                    tracked_person=tracked_person,
                    source_frame_shape=frame_bgr.shape,
                )
                if not display_window_initialized:
                    # WINDOW_NORMAL makes the Qt/OpenCV window user-resizable;
                    # the first size still follows --display-width/height.
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

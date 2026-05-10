import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from systems.shared.contracts.observation import decode_rgb_history_npz

PROJECT_ROOT = Path(__file__).resolve().parents[5]

from .capabilities import inspect_checkpoint_capabilities
from .policy_agent import NavDP_Agent


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=18888)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=str(PROJECT_ROOT / "artifacts" / "models" / "navdp-cross-modal.ckpt"),
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--tensorrt-mode",
    choices=("off", "auto", "required", "build"),
    default=os.environ.get("NAVDP_TENSORRT_MODE", "off"),
    help="Optional TensorRT mode for NavDP noise prediction: off, auto, required, or build.",
)
parser.add_argument(
    "--tensorrt-engine-dir",
    type=str,
    default=os.environ.get("NAVDP_TENSORRT_ENGINE_DIR", str(PROJECT_ROOT / "artifacts" / "models" / "navdp_tensorrt")),
    help="Directory containing or receiving NavDP TensorRT engine artifacts.",
)
parser.add_argument(
    "--tensorrt-precision",
    choices=("fp16", "fp32"),
    default=os.environ.get("NAVDP_TENSORRT_PRECISION", "fp16"),
    help="TensorRT builder precision for generated NavDP engines.",
)
parser.add_argument(
    "--save_debug_video",
    action="store_true",
    help="Save per-reset point-goal debug videos when imageio is available.",
)
parser.add_argument(
    "--debug_video_dir",
    type=str,
    default="",
    help="Optional output directory for debug videos. Defaults to the server working directory.",
)
args = parser.parse_known_args()[0]

app = Flask(__name__)
navdp_navigator = None
navdp_fps_writer = None
_debug_video_warning_emitted = False


def _checkpoint_capabilities():
    try:
        return inspect_checkpoint_capabilities(args.checkpoint)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to inspect NavDP checkpoint capabilities: {type(exc).__name__}: {exc}")
        return None


@app.route("/healthz", methods=["GET"])
def healthz():
    capabilities = _checkpoint_capabilities()
    return jsonify(
        {
            "status": "ok",
            "navigator_ready": navdp_navigator is not None,
            "supports_pixelgoal": bool(capabilities is not None and capabilities.supports_pixelgoal),
            "supports_imagegoal": bool(capabilities is not None and capabilities.supports_imagegoal),
            "tensorrt": (
                getattr(navdp_navigator, "tensorrt_status", None)
                if navdp_navigator is not None
                else {
                    "mode": args.tensorrt_mode,
                    "enabled": False,
                    "engine_dir": args.tensorrt_engine_dir,
                    "precision": args.tensorrt_precision,
                    "reason": "navigator_not_initialized",
                }
            ),
        }
    )


def _close_debug_writer():
    global navdp_fps_writer
    if navdp_fps_writer is None:
        return
    try:
        navdp_fps_writer.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to close NavDP debug video writer: {type(exc).__name__}: {exc}")
    finally:
        navdp_fps_writer = None


def _load_imageio():
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, exc
    return imageio, None


def _reset_debug_writer():
    global navdp_fps_writer, _debug_video_warning_emitted

    _close_debug_writer()
    if not args.save_debug_video:
        return

    imageio, exc = _load_imageio()
    if imageio is None:
        if not _debug_video_warning_emitted:
            print(
                f"[WARN] Debug video saving disabled because imageio is unavailable: {type(exc).__name__}: {exc}"
            )
            _debug_video_warning_emitted = True
        return

    output_dir = args.debug_video_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    format_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"{format_time}_fps_pointgoal.mp4")
    navdp_fps_writer = imageio.get_writer(output_path, fps=7)
    print(f"[INFO] Writing NavDP debug video to: {output_path}")


def _append_debug_frame(frame):
    if navdp_fps_writer is None:
        return
    try:
        navdp_fps_writer.append_data(frame)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to append NavDP debug video frame: {type(exc).__name__}: {exc}")
        _close_debug_writer()


def _require_navigator():
    if navdp_navigator is None:
        return jsonify({"error": "navigator_reset must be called before planner steps"}), 400
    return None


def _load_rgb_depth(batch_size):
    image_file = request.files["image"]
    depth_file = request.files["depth"]

    image = Image.open(image_file.stream)
    image = image.convert("RGB")
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert("I")
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))
    return image, depth


def _load_history_images(batch_size):
    history_file = request.files.get("history_npz")
    if history_file is None:
        return None
    if int(batch_size) != 1:
        raise ValueError("history_npz is only supported for NavDP batch_size=1.")
    payload = history_file.stream.read()
    history_rgb = decode_rgb_history_npz(payload)
    if history_rgb.size == 0:
        return history_rgb
    return np.ascontiguousarray(history_rgb[:, :, :, ::-1])


def _jsonify_plan(execute_trajectory, all_trajectory, all_values):
    return jsonify(
        {
            "trajectory": execute_trajectory.tolist(),
            "all_trajectory": all_trajectory.tolist(),
            "all_values": all_values.tolist(),
        }
    )


@app.route("/navigator_reset", methods=["POST"])
def navdp_reset():
    global navdp_navigator

    intrinsic = np.array(request.get_json().get("intrinsic"))
    threshold = float(request.get_json().get("stop_threshold"))
    batchsize = int(request.get_json().get("batch_size"))
    if navdp_navigator is None:
        navdp_navigator = NavDP_Agent(
            intrinsic,
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            navi_model=args.checkpoint,
            device=args.device,
            tensorrt_mode=args.tensorrt_mode,
            tensorrt_engine_dir=args.tensorrt_engine_dir,
            tensorrt_precision=args.tensorrt_precision,
        )
    navdp_navigator.reset(batchsize, threshold)
    _reset_debug_writer()
    capabilities = _checkpoint_capabilities()
    return jsonify(
        {
            "algo": "navdp",
            "supports_pixelgoal": bool(capabilities is not None and capabilities.supports_pixelgoal),
            "supports_imagegoal": bool(capabilities is not None and capabilities.supports_imagegoal),
            "tensorrt": getattr(navdp_navigator, "tensorrt_status", None),
        }
    )


@app.route("/navigator_reset_env", methods=["POST"])
def navdp_reset_env():
    precondition = _require_navigator()
    if precondition is not None:
        return precondition
    navdp_navigator.reset_env(int(request.get_json().get("env_id")))
    return jsonify({"algo": "navdp"})


@app.route("/pointgoal_step", methods=["POST"])
def navdp_step_xy():
    precondition = _require_navigator()
    if precondition is not None:
        return precondition

    start_time = time.time()
    goal_data = json.loads(request.form.get("goal_data"))
    goal_x = np.array(goal_data["goal_x"])
    goal_y = np.array(goal_data["goal_y"])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    batch_size = navdp_navigator.batch_size

    phase1_time = time.time()
    try:
        image, depth = _load_rgb_depth(batch_size)
        history_images = _load_history_images(batch_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pointgoal(
        goal,
        image,
        depth,
        history_images=history_images,
    )
    phase3_time = time.time()
    _append_debug_frame(trajectory_mask)
    phase4_time = time.time()
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )

    return _jsonify_plan(execute_trajectory, all_trajectory, all_values)


@app.route("/pixelgoal_step", methods=["POST"])
def navdp_step_pixel():
    precondition = _require_navigator()
    if precondition is not None:
        return precondition

    start_time = time.time()
    goal_data = json.loads(request.form.get("goal_data"))
    goal_x = np.array(goal_data["goal_x"])
    goal_y = np.array(goal_data["goal_y"])
    goal = np.stack((goal_x, goal_y), axis=1)
    batch_size = navdp_navigator.batch_size

    phase1_time = time.time()
    try:
        image, depth = _load_rgb_depth(batch_size)
        history_images = _load_history_images(batch_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_pixelgoal(
        goal,
        image,
        depth,
        history_images=history_images,
    )
    phase3_time = time.time()
    _append_debug_frame(trajectory_mask)
    phase4_time = time.time()
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )
    return _jsonify_plan(execute_trajectory, all_trajectory, all_values)


@app.route("/imagegoal_step", methods=["POST"])
def navdp_step_image():
    precondition = _require_navigator()
    if precondition is not None:
        return precondition

    start_time = time.time()
    goal_file = request.files["goal"]
    batch_size = navdp_navigator.batch_size

    phase1_time = time.time()
    try:
        image, depth = _load_rgb_depth(batch_size)
        history_images = _load_history_images(batch_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    goal = Image.open(goal_file.stream)
    goal = goal.convert("RGB")
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))

    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_imagegoal(
        goal,
        image,
        depth,
        history_images=history_images,
    )
    phase3_time = time.time()
    _append_debug_frame(trajectory_mask)
    phase4_time = time.time()
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )
    return _jsonify_plan(execute_trajectory, all_trajectory, all_values)


@app.route("/nogoal_step", methods=["POST"])
def navdp_step_nogoal():
    precondition = _require_navigator()
    if precondition is not None:
        return precondition

    start_time = time.time()
    batch_size = navdp_navigator.batch_size

    phase1_time = time.time()
    try:
        image, depth = _load_rgb_depth(batch_size)
        history_images = _load_history_images(batch_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_nogoal(
        image,
        depth,
        history_images=history_images,
    )
    phase3_time = time.time()
    _append_debug_frame(trajectory_mask)
    phase4_time = time.time()
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )
    return _jsonify_plan(execute_trajectory, all_trajectory, all_values)


@app.route("/navdp_step_ip_mixgoal", methods=["POST"])
def navdp_step_ip_mixgoal():
    precondition = _require_navigator()
    if precondition is not None:
        return precondition

    start_time = time.time()
    batch_size = navdp_navigator.batch_size

    point_goal_data = json.loads(request.form.get("goal_data"))
    point_goal_x = np.array(point_goal_data["goal_x"])
    point_goal_y = np.array(point_goal_data["goal_y"])
    point_goal = np.stack((point_goal_x, point_goal_y, np.zeros_like(point_goal_x)), axis=1)

    image_goal_file = request.files["image_goal"]
    image_goal = Image.open(image_goal_file.stream)
    image_goal = image_goal.convert("RGB")
    image_goal = np.asarray(image_goal)
    image_goal = cv2.cvtColor(image_goal, cv2.COLOR_RGB2BGR)
    image_goal = image_goal.reshape((batch_size, -1, image_goal.shape[1], 3))

    phase1_time = time.time()
    try:
        image, depth = _load_rgb_depth(batch_size)
        history_images = _load_history_images(batch_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = navdp_navigator.step_point_image_goal(
        point_goal,
        image_goal,
        image,
        depth,
        history_images=history_images,
    )
    phase3_time = time.time()
    _append_debug_frame(trajectory_mask)
    phase4_time = time.time()
    print(
        "phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f"
        % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time,
        )
    )
    return _jsonify_plan(execute_trajectory, all_trajectory, all_values)


def main() -> int:
    print(f"[INFO] NavDP server listening on 127.0.0.1:{args.port}")
    print(f"[INFO] Checkpoint        : {args.checkpoint}")
    print(f"[INFO] Device            : {args.device}")
    print(f"[INFO] TensorRT mode    : {args.tensorrt_mode}")
    print(f"[INFO] TensorRT engines : {args.tensorrt_engine_dir}")
    print(f"[INFO] TensorRT precision: {args.tensorrt_precision}")
    print(f"[INFO] Save debug video  : {args.save_debug_video}")
    if args.debug_video_dir:
        print(f"[INFO] Debug video dir   : {args.debug_video_dir}")
    app.run(host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

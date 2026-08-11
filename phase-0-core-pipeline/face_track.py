"""
Face tracking stage: detects the speaking face in a clip using MediaPipe's
FaceDetector Tasks API (the only face API available in the installed
mediapipe==1.0.0 -- mediapipe.solutions does not exist in this version)
and produces a smoothed sequence of crop centers for reframe.py to apply.
Falls back to a static center-crop if no face is detected anywhere in
the clip.
"""

import logging
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

logger = logging.getLogger(__name__)

# Google's public MediaPipe model storage -- free, no API key, no paid tier.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
# Cached outside the repo (like Whisper's own weight cache), so it never
# needs a .gitignore entry or gets committed.
MODEL_CACHE_DIR = Path.home() / ".cache" / "reel-generator" / "models"
MODEL_PATH = MODEL_CACHE_DIR / "blaze_face_short_range.tflite"

# Detect roughly every 200ms of video -- plenty for smooth crop movement,
# cheap enough to run on CPU.
SAMPLE_INTERVAL_SEC = 0.2
MIN_DETECTION_CONFIDENCE = 0.5

# Exponential moving average smoothing factor for crop centers -- higher
# = more responsive/jittery, lower = smoother/laggier.
EMA_ALPHA = 0.18

# Target 9:16 vertical aspect ratio for the crop box.
TARGET_ASPECT_W = 9
TARGET_ASPECT_H = 16


def track_face(clip_path: Path) -> dict:
    """
    Analyze `clip_path` and return crop guidance for reframe.py:
        {
          "face_detected": bool,
          "source_width": int, "source_height": int,
          "crop_width": int, "crop_height": int,
          "centers": [{"t": float, "cx": int, "cy": int}, ...],
        }
    `centers` has one entry (the whole-clip fallback center) when no face
    was detected, or one entry per sampled+smoothed timestamp otherwise.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip for face tracking: {clip_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        crop_width, crop_height = _crop_dimensions(width, height)

        raw_centers = _detect_face_centers(cap, fps, width, height)
    finally:
        cap.release()

    if not raw_centers:
        logger.info("No face detected in %s -- using center-crop fallback.", clip_path)
        cx, cy = _clamp_center(width / 2, height / 2, width, height, crop_width, crop_height)
        return {
            "face_detected": False,
            "source_width": width,
            "source_height": height,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "centers": [{"t": 0.0, "cx": cx, "cy": cy}],
        }

    smoothed = _smooth_centers(raw_centers)
    centers = []
    for t, cx, cy in smoothed:
        cx, cy = _clamp_center(cx, cy, width, height, crop_width, crop_height)
        centers.append({"t": t, "cx": cx, "cy": cy})

    return {
        "face_detected": True,
        "source_width": width,
        "source_height": height,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "centers": centers,
    }


def _crop_dimensions(width: int, height: int) -> tuple[int, int]:
    """9:16 crop box that fits inside the source frame."""
    target_ratio = TARGET_ASPECT_W / TARGET_ASPECT_H
    if width / height > target_ratio:
        crop_height = height
        crop_width = int(round(height * target_ratio))
    else:
        crop_width = width
        crop_height = int(round(width / target_ratio))
    # Even dimensions are required for yuv420p encoding.
    crop_width -= crop_width % 2
    crop_height -= crop_height % 2
    return crop_width, crop_height


def _clamp_center(cx: float, cy: float, width: int, height: int,
                   crop_width: int, crop_height: int) -> tuple[int, int]:
    half_w, half_h = crop_width / 2, crop_height / 2
    cx = min(max(cx, half_w), width - half_w)
    cy = min(max(cy, half_h), height - half_h)
    return int(round(cx)), int(round(cy))


def _detect_face_centers(cap: cv2.VideoCapture, fps: float, width: int, height: int) -> list[tuple[float, float, float]]:
    detector = _load_detector()
    centers: list[tuple[float, float, float]] = []
    frame_interval = max(int(round(fps * SAMPLE_INTERVAL_SEC)), 1)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp_ms = int((frame_idx / fps) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, timestamp_ms)
                center = _pick_face_center(result, width, height)
                if center is not None:
                    centers.append((timestamp_ms / 1000.0, center[0], center[1]))
            frame_idx += 1
    finally:
        detector.close()

    return centers


def _load_detector() -> FaceDetector:
    _ensure_model_downloaded()
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionTaskRunningMode.VIDEO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    return FaceDetector.create_from_options(options)


def _ensure_model_downloaded() -> None:
    if MODEL_PATH.exists():
        return
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face detection model to %s ...", MODEL_PATH)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _pick_face_center(result, width: int, height: int):
    """Largest bounding box wins (proxy for the primary/closest speaker);
    ties broken by proximity to frame center."""
    if not result.detections:
        return None

    def area(detection):
        bb = detection.bounding_box
        return bb.width * bb.height

    def centrality(detection):
        bb = detection.bounding_box
        cx = bb.origin_x + bb.width / 2
        cy = bb.origin_y + bb.height / 2
        return -((cx - width / 2) ** 2 + (cy - height / 2) ** 2)

    best = max(result.detections, key=lambda d: (area(d), centrality(d)))
    bb = best.bounding_box
    return bb.origin_x + bb.width / 2, bb.origin_y + bb.height / 2


def _smooth_centers(raw_centers: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    smoothed = []
    ema_cx, ema_cy = None, None
    for t, cx, cy in raw_centers:
        if ema_cx is None:
            ema_cx, ema_cy = cx, cy
        else:
            ema_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * ema_cx
            ema_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * ema_cy
        smoothed.append((t, ema_cx, ema_cy))
    return smoothed

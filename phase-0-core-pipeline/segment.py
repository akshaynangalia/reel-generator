"""
Segmentation stage: expands/clamps a highlight window to fit the
30-60s clip bound, snaps its edges to transcript word boundaries so cuts
never land mid-word, and cuts the resulting clip from the source video
via ffmpeg.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_CLIP_SEC = 30.0
MAX_CLIP_SEC = 60.0

# How close a computed boundary must be to a word start/end to snap to it.
WORD_SNAP_TOLERANCE_SEC = 1.0


def build_clip_bounds(window: dict, transcript: dict, video_duration: float) -> tuple[float, float]:
    """
    Expand or clamp `window` (a highlight window dict with start/end/
    segment_scores from highlight_detection.detect_highlights) to fit
    within [MIN_CLIP_SEC, MAX_CLIP_SEC], then snap both edges to the
    nearest transcript word boundary.

    Assumes the caller has already verified video_duration >= MIN_CLIP_SEC
    (checked once, up front, in pipeline.py).
    """
    start, end = window["start"], window["end"]
    natural_len = end - start

    if natural_len < MIN_CLIP_SEC:
        start, end = _expand_symmetric(start, end, MIN_CLIP_SEC, video_duration)
    elif natural_len > MAX_CLIP_SEC:
        start, end = _center_on_peak(window, transcript, video_duration)

    words = _flatten_words(transcript)
    snapped_start = max(_snap_to_word_boundary(start, words), 0.0)
    snapped_end = min(_snap_to_word_boundary(end, words), video_duration)

    # Each edge is snapped independently, so in rare cases both can move
    # inward at once and pull the clip under MIN_CLIP_SEC (or, less
    # commonly, outward past MAX_CLIP_SEC). Revert whichever single edge
    # fixes the violation before falling back to reverting both -- this
    # guarantees the pre-snap bound (already valid by construction) is
    # never broken by snapping.
    snapped_len = snapped_end - snapped_start
    if snapped_len < MIN_CLIP_SEC:
        if snapped_end - start >= MIN_CLIP_SEC:
            snapped_start = start
        elif end - snapped_start >= MIN_CLIP_SEC:
            snapped_end = end
        else:
            snapped_start, snapped_end = start, end
    elif snapped_len > MAX_CLIP_SEC:
        if end - snapped_start <= MAX_CLIP_SEC:
            snapped_end = end
        elif snapped_end - start <= MAX_CLIP_SEC:
            snapped_start = start
        else:
            snapped_start, snapped_end = start, end

    return snapped_start, snapped_end


def cut_clip(video_path: Path, start: float, end: float, output_path: Path) -> None:
    """Cut [start, end] from video_path into output_path. Video is
    re-encoded for frame-accurate cuts; audio is stream-copied per the
    project's never-touch-the-audio rule."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        str(output_path),
    ]
    logger.info("Cutting clip [%.2fs, %.2fs] -> %s", start, end, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to cut clip {output_path}:\n{result.stderr}")


def _expand_symmetric(start: float, end: float, target_len: float, video_duration: float) -> tuple[float, float]:
    deficit = target_len - (end - start)
    half = deficit / 2
    new_start = start - half
    new_end = end + half

    if new_start < 0:
        overflow = -new_start
        new_start = 0.0
        new_end += overflow
    if new_end > video_duration:
        overflow = new_end - video_duration
        new_end = video_duration
        new_start -= overflow

    return max(new_start, 0.0), min(new_end, video_duration)


def _center_on_peak(window: dict, transcript: dict, video_duration: float) -> tuple[float, float]:
    """Window is longer than MAX_CLIP_SEC -- center a MAX_CLIP_SEC window
    on the highest-scoring member segment's midpoint, rather than just
    truncating from one end."""
    segments_by_id = {seg["id"]: seg for seg in transcript["segments"]}
    peak_id = max(window["segment_scores"], key=window["segment_scores"].get)
    peak_segment = segments_by_id[peak_id]
    peak_mid = (peak_segment["start"] + peak_segment["end"]) / 2

    new_start = peak_mid - MAX_CLIP_SEC / 2
    new_end = peak_mid + MAX_CLIP_SEC / 2

    # Stay within the original highlight window first...
    new_start = max(new_start, window["start"])
    new_end = min(new_end, window["end"])

    # ...then within the video itself, shifting rather than just clamping
    # so the clip keeps its full length where possible.
    if new_start < 0:
        shift = -new_start
        new_start = 0.0
        new_end = min(new_end + shift, video_duration)
    if new_end > video_duration:
        shift = new_end - video_duration
        new_end = video_duration
        new_start = max(new_start - shift, 0.0)

    return new_start, new_end


def _flatten_words(transcript: dict) -> list[dict]:
    words = []
    for seg in transcript["segments"]:
        words.extend(seg.get("words", []))
    words.sort(key=lambda w: w["start"])
    return words


def _snap_to_word_boundary(time: float, words: list[dict]) -> float:
    if not words:
        return time
    boundaries = [w["start"] for w in words] + [w["end"] for w in words]
    candidates = [b for b in boundaries if abs(b - time) <= WORD_SNAP_TOLERANCE_SEC]
    if not candidates:
        return time
    return min(candidates, key=lambda b: abs(b - time))

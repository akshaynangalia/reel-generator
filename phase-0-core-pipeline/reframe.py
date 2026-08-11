"""
Reframe stage: crops a clip to 9:16 vertical using the crop centers from
face_track.py. A face-tracked clip gets a moving crop driven by ffmpeg's
`sendcmd` filter (one command per sampled/smoothed timestamp); the no-face
fallback uses a single static crop for the whole clip. Audio is always
stream-copied, never re-encoded.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920


def reframe_clip(clip_path: Path, track: dict, output_path: Path) -> None:
    """
    Crop `clip_path` to 9:16 vertical per `track` (the dict returned by
    face_track.track_face) and write the result to output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent

    crop_width = track["crop_width"]
    crop_height = track["crop_height"]
    centers = track["centers"]

    if track["face_detected"] and len(centers) > 1:
        cmds_path = work_dir / "crop_cmds.txt"
        _write_sendcmd_file(centers, crop_width, crop_height, cmds_path)
        x0 = centers[0]["cx"] - crop_width // 2
        y0 = centers[0]["cy"] - crop_height // 2
        filter_complex = (
            f"[0:v]sendcmd=f={cmds_path.name},"
            f"crop@c=w={crop_width}:h={crop_height}:x={x0}:y={y0},"
            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}[v]"
        )
    else:
        center = centers[0]
        x0 = center["cx"] - crop_width // 2
        y0 = center["cy"] - crop_height // 2
        filter_complex = (
            f"[0:v]crop={crop_width}:{crop_height}:{x0}:{y0},"
            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}[v]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path.resolve()),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        output_path.name,
    ]
    logger.info(
        "Reframing %s -> %s (face_detected=%s)",
        clip_path, output_path, track["face_detected"],
    )
    # Run with cwd=work_dir so the sendcmd filename and output filename
    # can stay bare, sidestepping ":"/"\" escaping of Windows paths inside
    # the filtergraph string (the same issue captions.py works around).
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to reframe clip {output_path}:\n{result.stderr}")


def _write_sendcmd_file(centers: list[dict], crop_width: int, crop_height: int, cmds_path: Path) -> None:
    lines = []
    for c in centers:
        x = c["cx"] - crop_width // 2
        y = c["cy"] - crop_height // 2
        lines.append(f"{c['t']:.3f} crop@c x '{x}', crop@c y '{y}';")
    cmds_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

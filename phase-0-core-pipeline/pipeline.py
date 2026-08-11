"""
Pipeline orchestrator / CLI entry point.

Runs the full Phase 0 pipeline against a single input video:
  validate input -> duration probe -> transcribe -> highlight detection ->
  per highlight: segment -> face track -> reframe -> captions -> final.mp4

Usage:
    python pipeline.py <input_video> [--output-dir DIR] [--model-size base]
                        [--max-clips 3] [--force-retranscribe]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import captions
import face_track
import highlight_detection
import reframe
import segment
import transcribe
from errors import InputTooShortError, PipelineError

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _setup_logging()

    input_video = Path(args.input_video)
    output_root = Path(args.output_dir)

    try:
        if not input_video.exists():
            raise PipelineError(f"Input video not found: {input_video}")

        duration = _probe_duration(input_video)
        if duration < segment.MIN_CLIP_SEC:
            raise InputTooShortError(
                f"Input video is {duration:.1f}s long, shorter than the "
                f"{segment.MIN_CLIP_SEC:.0f}s minimum clip length -- nothing to do."
            )

        video_output_dir = output_root / input_video.stem
        transcript = transcribe.transcribe_video(
            input_video,
            video_output_dir,
            model_size=args.model_size,
            force_retranscribe=args.force_retranscribe,
        )

        highlights = highlight_detection.detect_highlights(
            transcript, input_video, max_clips=args.max_clips
        )
    except PipelineError as e:
        logger.error("Pipeline stopped: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error before clip generation began.")
        sys.exit(1)

    produced = 0
    for i, window in enumerate(highlights, start=1):
        try:
            _process_clip(i, window, transcript, input_video, video_output_dir, duration)
            produced += 1
        except Exception:
            logger.exception("Skipping clip %d due to an unexpected error.", i)

    if produced == 0:
        logger.error("No clips were successfully produced.")
        sys.exit(1)

    logger.info(
        "Done. %d/%d clip(s) produced in %s", produced, len(highlights), video_output_dir
    )


def _process_clip(
    index: int,
    window: dict,
    transcript: dict,
    input_video: Path,
    video_output_dir: Path,
    video_duration: float,
) -> None:
    start, end = segment.build_clip_bounds(window, transcript, video_duration)
    clip_dir = video_output_dir / f"clip_{index:02d}_{start:.0f}-{end:.0f}s"
    clip_dir.mkdir(parents=True, exist_ok=True)

    raw_path = clip_dir / "clip_raw.mp4"
    segment.cut_clip(input_video, start, end, raw_path)

    track = face_track.track_face(raw_path)

    vertical_path = clip_dir / "clip_vertical.mp4"
    reframe.reframe_clip(raw_path, track, vertical_path)

    ass_path = clip_dir / "captions.ass"
    captions.build_captions(transcript, start, end, ass_path)

    final_path = clip_dir / "final.mp4"
    captions.burn_in_captions(vertical_path, ass_path, final_path)

    logger.info("Clip %d complete: %s [%.1fs - %.1fs]", index, final_path, start, end)


def _probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise PipelineError(f"ffprobe could not read duration for {video_path}:\n{result.stderr}")
    return float(result.stdout.strip())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a raw video into captioned, vertical, face-tracked highlight clips."
    )
    parser.add_argument("input_video", help="Path to the source video file.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write clips into (default: phase-0-core-pipeline/output).",
    )
    parser.add_argument(
        "--model-size",
        default="base",
        help="Whisper model size (default: base -- tuned for CPU-only hardware).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=3,
        help="Maximum number of highlight clips to produce (default: 3).",
    )
    parser.add_argument(
        "--force-retranscribe",
        action="store_true",
        help="Ignore any cached transcript and re-run Whisper.",
    )
    return parser.parse_args(argv)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    main()

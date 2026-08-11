"""
Captions stage: converts a clip's word-level transcript slice into styled
ASS subtitles (needed for real styling -- SRT can't do this) and burns
them into the clip via ffmpeg's `ass` filter (libass).
"""

import logging
import subprocess
from pathlib import Path

from reframe import OUTPUT_WIDTH, OUTPUT_HEIGHT

logger = logging.getLogger(__name__)

MAX_CHARS_PER_LINE = 30
MAX_WORDS_PER_LINE = 6
PAUSE_BREAK_SEC = 0.3

# A font that ships with Windows -- no bundled/downloaded font asset needed.
FONT_NAME = "Arial Black"
FONT_SIZE = 64
PRIMARY_COLOUR = "&H00FFFFFF"  # ASS order &HAABBGGRR -- opaque white fill
OUTLINE_COLOUR = "&H00000000"  # opaque black outline, for contrast on any background
OUTLINE_WIDTH = 3
# Keeps captions inside the bottom-safe third of a 9:16 frame, clear of
# typical platform UI overlap zones.
MARGIN_V = 220


def build_captions(transcript: dict, clip_start: float, clip_end: float, ass_path: Path) -> None:
    """
    Slice `transcript`'s word timestamps to [clip_start, clip_end], chunk
    them into short readable lines, and write an ASS file at ass_path
    with timestamps relative to the clip (t=0 == clip_start).
    """
    words = _words_in_range(transcript, clip_start, clip_end)
    chunks = _chunk_words(words)
    _write_ass(chunks, ass_path)


def burn_in_captions(clip_path: Path, ass_path: Path, output_path: Path) -> None:
    """Burn ass_path's subtitles into clip_path, writing output_path.
    ass_path must live in the same directory as output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path.resolve()),
        "-vf", f"ass={ass_path.name}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "copy",
        output_path.name,
    ]
    logger.info("Burning in captions %s -> %s", clip_path, output_path)
    # Run with cwd=work_dir and a bare filename for the ass= argument --
    # ffmpeg's filtergraph parser treats ":" as an option separator, which
    # collides with Windows absolute paths (e.g. "C:\...") if used directly.
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to burn in captions for {output_path}:\n{result.stderr}")


def _words_in_range(transcript: dict, clip_start: float, clip_end: float) -> list[dict]:
    words = []
    for seg in transcript["segments"]:
        for w in seg.get("words", []):
            if w["end"] < clip_start or w["start"] > clip_end:
                continue
            words.append(
                {
                    "word": w["word"],
                    "start": max(w["start"], clip_start) - clip_start,
                    "end": min(w["end"], clip_end) - clip_start,
                }
            )
    words.sort(key=lambda w: w["start"])
    return words


def _chunk_words(words: list[dict]) -> list[list[dict]]:
    """Group words into short caption lines, preferring to break at a
    natural pause over a hard character/word cutoff."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    prev_end = None

    for w in words:
        word_text = w["word"].strip()
        if not word_text:
            continue
        gap = (w["start"] - prev_end) if prev_end is not None else 0.0

        should_break = current and (
            gap > PAUSE_BREAK_SEC
            or len(current) >= MAX_WORDS_PER_LINE
            or current_len + len(word_text) + 1 > MAX_CHARS_PER_LINE
        )
        if should_break:
            chunks.append(current)
            current = []
            current_len = 0

        current.append(w)
        current_len += len(word_text) + 1
        prev_end = w["end"]

    if current:
        chunks.append(current)

    return chunks


def _write_ass(chunks: list[list[dict]], ass_path: Path) -> None:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {OUTPUT_WIDTH}\n"
        f"PlayResY: {OUTPUT_HEIGHT}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_NAME},{FONT_SIZE},{PRIMARY_COLOUR},&H000000FF,"
        f"{OUTLINE_COLOUR},&H00000000,-1,0,0,0,100,100,0,0,1,{OUTLINE_WIDTH},0,"
        f"2,40,40,{MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for i, chunk in enumerate(chunks):
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        if i + 1 < len(chunks):
            end = min(end, chunks[i + 1][0]["start"])
        if end <= start:
            end = start + 0.5
        text = " ".join(w["word"].strip() for w in chunk)
        lines.append(f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},Default,,0,0,0,,{text}\n")

    ass_path.write_text("".join(lines), encoding="utf-8")


def _fmt_ts(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.CS (centiseconds)."""
    seconds = max(seconds, 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

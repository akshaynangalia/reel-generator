"""
Transcription stage: runs local Whisper on the input video's audio track
and produces a timestamped transcript, cached as JSON so repeated runs
against the same video are both fast and deterministic.
"""

import json
import logging
from pathlib import Path

import whisper

from errors import NoSpeechDetectedError

logger = logging.getLogger(__name__)

# Segments with no_speech_prob above this are treated as silence when
# deciding whether the video had any detectable speech at all.
NO_SPEECH_PROB_THRESHOLD = 0.9

# Whisper is known to hallucinate a stray word or two (commonly "you",
# "thank you") on pure silence/near-silence rather than returning zero
# segments -- observed directly during Phase 0 testing on a silent clip,
# where only one of two hallucinated segments cleared
# NO_SPEECH_PROB_THRESHOLD. A minimum total word count catches this
# regardless of individual no_speech_prob values.
MIN_WORD_COUNT = 5


def transcribe_video(
    video_path: Path,
    output_dir: Path,
    model_size: str = "base",
    force_retranscribe: bool = False,
) -> dict:
    """
    Transcribe `video_path` with Whisper and return the transcript dict.

    Results are cached at `output_dir/transcript.json`. If that file
    already exists and is newer than the input video, it is reused
    instead of re-running Whisper. This is what keeps repeated pipeline
    runs against the same input deterministic (and much faster).

    Raises NoSpeechDetectedError if the video has no usable speech.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.json"

    if not force_retranscribe and _cache_is_fresh(transcript_path, video_path):
        logger.info("Using cached transcript at %s", transcript_path)
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        _raise_if_no_speech(transcript)
        return transcript

    logger.info(
        "Transcribing %s with Whisper model '%s' (this can take a while on CPU)...",
        video_path,
        model_size,
    )
    model = whisper.load_model(model_size)
    result = model.transcribe(
        str(video_path),
        fp16=False,  # CPU-only hardware -- avoids Whisper's fp16 warning spam
        word_timestamps=True,
        verbose=False,
    )

    transcript = _build_transcript(video_path, model_size, result)

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info("Transcript written to %s", transcript_path)

    _raise_if_no_speech(transcript)
    return transcript


def _cache_is_fresh(transcript_path: Path, video_path: Path) -> bool:
    if not transcript_path.exists():
        return False
    return transcript_path.stat().st_mtime >= video_path.stat().st_mtime


def _build_transcript(video_path: Path, model_size: str, result: dict) -> dict:
    segments = []
    for i, seg in enumerate(result.get("segments", [])):
        words = [
            {
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "probability": w.get("probability", 0.0),
            }
            for w in seg.get("words", [])
        ]
        segments.append(
            {
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "no_speech_prob": seg.get("no_speech_prob", 0.0),
                "avg_logprob": seg.get("avg_logprob", 0.0),
                "words": words,
            }
        )

    duration = segments[-1]["end"] if segments else 0.0

    return {
        "input_video": str(video_path),
        "model_size": model_size,
        "language": result.get("language", "unknown"),
        "duration_sec": duration,
        "segments": segments,
    }


def _raise_if_no_speech(transcript: dict) -> None:
    segments = transcript.get("segments", [])
    if not segments:
        raise NoSpeechDetectedError("Whisper found no speech segments in the input video.")
    if all(seg.get("no_speech_prob", 0.0) > NO_SPEECH_PROB_THRESHOLD for seg in segments):
        raise NoSpeechDetectedError(
            "Whisper detected only silence/non-speech audio in the input video."
        )
    total_words = sum(len(seg.get("words", [])) for seg in segments)
    if total_words < MIN_WORD_COUNT:
        raise NoSpeechDetectedError(
            f"Whisper found only {total_words} word(s) in the input video -- "
            "likely silence (Whisper can hallucinate a stray word or two on "
            "pure silence rather than returning nothing)."
        )

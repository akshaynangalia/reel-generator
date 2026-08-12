"""
Transcription stage: runs local Whisper on the input video's audio track
and produces a timestamped transcript, cached as JSON so repeated runs
against the same video are both fast and deterministic.
"""

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import whisper
import whisper.audio as whisper_audio

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

# How much audio to sample for automatic language detection, in seconds.
# Matches Whisper's own internal auto-detect window ("first 30 seconds"),
# so detection quality matches today's full-clip auto-detect rather than
# being degraded by an arbitrarily short custom sample. Decoding only
# this much audio up front (instead of reusing whisper.audio.load_audio,
# which has no duration limit) avoids fully decoding a long video twice --
# once for detection, once inside model.transcribe()'s own internal load.
LANGUAGE_SAMPLE_SEC = 30.0

# Used when the caller doesn't request a specific model size.
DEFAULT_MODEL_SIZE = "base"

# Auto-escalation target when detected/pinned language isn't English and
# the caller didn't request a specific model size. Testing found that
# "base" reliably romanizes/garbles non-Latin-script languages (e.g.
# Hindi) even when the language is correctly identified -- explicit
# language pinning alone doesn't fix that, since Whisper's own internal
# auto-detect was already getting the language right; it's "base"'s
# decoder that's weak at generating native script. "small" was confirmed
# (on a real Hindi test video) to produce correct Devanagari.
ESCALATED_MODEL_SIZE = "small"


def transcribe_video(
    video_path: Path,
    output_dir: Path,
    model_size: str | None = None,
    force_retranscribe: bool = False,
    language: str | None = None,
) -> dict:
    """
    Transcribe `video_path` with Whisper and return the transcript dict.

    Results are cached at `output_dir/transcript.json`. If that file
    already exists and is newer than the input video, it is reused
    instead of re-running Whisper. This is what keeps repeated pipeline
    runs against the same input deterministic (and much faster).

    `model_size`, if None (the default), starts at DEFAULT_MODEL_SIZE and
    auto-escalates to ESCALATED_MODEL_SIZE if the resolved language isn't
    English -- see ESCALATED_MODEL_SIZE's comment for why. An explicit
    `model_size` is always honored as-is, with no escalation.

    `language` pins the Whisper language code (e.g. "en", "hi") used for
    the full transcription. If None (the default), it's auto-detected
    from a short audio sample first, then pinned for the full
    transcription -- this is more reliable than letting model.transcribe
    auto-detect internally over the whole clip.

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

    auto_select_model = model_size is None
    initial_size = model_size if model_size is not None else DEFAULT_MODEL_SIZE
    logger.info(
        "Transcribing %s with Whisper model '%s' (this can take a while on CPU)...",
        video_path,
        initial_size,
    )
    model = whisper.load_model(initial_size)
    pinned_language = language if language is not None else _detect_language(model, video_path)

    used_size = initial_size
    if auto_select_model and pinned_language != "en" and initial_size != ESCALATED_MODEL_SIZE:
        logger.info(
            "Detected non-English language '%s' -- escalating Whisper model "
            "'%s' -> '%s' for better native-script transcription quality.",
            pinned_language, initial_size, ESCALATED_MODEL_SIZE,
        )
        model = whisper.load_model(ESCALATED_MODEL_SIZE)
        used_size = ESCALATED_MODEL_SIZE

    result = model.transcribe(
        str(video_path),
        fp16=False,  # CPU-only hardware -- avoids Whisper's fp16 warning spam
        word_timestamps=True,
        verbose=False,
        language=pinned_language,
    )

    transcript = _build_transcript(video_path, used_size, result)

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info("Transcript written to %s", transcript_path)

    _raise_if_no_speech(transcript)
    return transcript


def _detect_language(model, video_path: Path) -> str:
    """Detect the spoken language from a short audio sample and return
    its Whisper language code, so the full transcription can be pinned
    to it instead of relying on Whisper's own whole-clip auto-detect."""
    if not model.is_multilingual:
        return "en"
    sample = _load_audio_sample(video_path, whisper_audio.SAMPLE_RATE, LANGUAGE_SAMPLE_SEC)
    mel = whisper_audio.log_mel_spectrogram(sample, model.dims.n_mels, padding=whisper_audio.N_SAMPLES)
    mel_segment = whisper_audio.pad_or_trim(mel, whisper_audio.N_FRAMES).to(model.device)
    _, probs = model.detect_language(mel_segment)
    detected = max(probs, key=probs.get)
    logger.info("Auto-detected language: %s (p=%.2f)", detected, probs[detected])
    return detected


def _load_audio_sample(video_path: Path, sr: int, duration_sec: float) -> np.ndarray:
    """Decode only the first `duration_sec` seconds of `video_path`'s
    audio track to a mono float32 waveform at sample rate `sr`, mirroring
    whisper.audio.load_audio's ffmpeg invocation but bounded by `-t` so a
    long video isn't fully decoded twice (once here, once inside
    model.transcribe()'s own internal load)."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-t", str(duration_sec),
        "-i", str(video_path),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to sample audio from {video_path} for language "
            f"detection:\n{result.stderr.decode(errors='replace')}"
        )
    return np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0


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

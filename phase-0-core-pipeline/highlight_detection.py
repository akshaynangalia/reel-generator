"""
Highlight detection stage: scores each transcript segment on a
deterministic keyword / audio-energy / pause / speech-rate heuristic (no
paid AI model), merges high-scoring runs of segments into candidate
highlight windows, and selects the top-scoring windows.

Everything here is pure arithmetic over the cached transcript and a
deterministic decode of the source audio -- no randomness, no model
inference -- so the same input always produces the same highlights.
"""

import logging
import re
import statistics
from pathlib import Path

import numpy as np
import whisper.audio as whisper_audio

from errors import NoHighlightFoundError

logger = logging.getLogger(__name__)

# Small curated list of words that tend to mark an emphatic / highlight-
# worthy moment in spoken content. Intentionally simple word matching --
# no external model.
HIGHLIGHT_KEYWORDS = {
    "amazing", "incredible", "insane", "crazy", "best", "worst", "never",
    "huge", "unbelievable", "literally", "shocking", "wow", "seriously",
    "honestly", "secret", "mistake", "wrong", "hate", "love", "epic",
    "ridiculous", "wild", "brutal", "genius", "terrible", "perfect",
}

# Same idea as HIGHLIGHT_KEYWORDS, in Devanagari-script Hindi. Checked
# together with the English list on every segment (never switched
# per-video) so mixed Hindi-English "Hinglish" speech still scores hits
# from whichever language a given phrase happens to use.
HIGHLIGHT_KEYWORDS_HI = {
    "अद्भुत", "अविश्वसनीय", "शानदार", "बेहतरीन", "जबरदस्त", "कमाल",
    "सनसनीखेज", "हैरान", "पागल", "विशाल", "सच", "राज़", "गलती",
    "गलत", "नफरत", "प्यार", "बेकार", "बर्बाद", "वाह",
}

ALL_KEYWORDS = HIGHLIGHT_KEYWORDS | HIGHLIGHT_KEYWORDS_HI

SAMPLE_RATE = 16000
ENERGY_FRAME_MS = 20

# Signal weights -- easy to retune without touching logic. Keyword gets a
# higher weight than the others: explicit exclamatory language is the most
# direct, reliable signal of an "obvious" highlight, and testing showed an
# equal weighting let a strong keyword hit get diluted below the selection
# threshold by comparatively noisy pause/rate signals on flat/monotone
# audio (e.g. TTS-narrated or deadpan-delivered speech).
W_KEYWORD = 1.5
W_ENERGY = 1.0
W_PAUSE = 1.0
W_RATE = 1.0

# Segments merge into the same highlight window if the gap between them
# is no more than this many seconds.
MERGE_GAP_SEC = 1.5

# How many standard deviations above the mean a window's score must clear
# to qualify as a highlight.
THRESHOLD_STDEV_MULT = 1.0


def detect_highlights(transcript: dict, video_path: Path) -> list[dict]:
    """
    Score transcript segments, merge them into candidate highlight
    windows, and return ALL qualifying windows (score-sorted,
    non-overlapping) as:
        {"start": float, "end": float, "score": float,
         "segment_ids": [int, ...], "segment_scores": {id: float, ...}}

    Capping how many of these are actually turned into clips is the
    caller's job (see pipeline.py's _select_non_overlapping_clips) --
    it happens *after* 30-60s clip-bound expansion, since two windows
    that don't overlap here can still expand into each other's space.

    Raises NoHighlightFoundError if nothing clears the selection threshold.
    """
    segments = transcript["segments"]
    if not segments:
        raise NoHighlightFoundError("Transcript has no segments to score.")

    keyword_scores = [_keyword_score(seg["text"]) for seg in segments]
    energy_scores = _energy_scores(video_path, segments)
    pause_scores = _pause_scores(segments)
    rate_scores = [_rate_score(seg) for seg in segments]

    composite = [
        W_KEYWORD * kw + W_ENERGY * en + W_PAUSE * pa + W_RATE * ra
        for kw, en, pa, ra in zip(
            _zscore(keyword_scores),
            _zscore(energy_scores),
            _zscore(pause_scores),
            _zscore(rate_scores),
        )
    ]
    segment_scores_by_id = {seg["id"]: score for seg, score in zip(segments, composite)}

    windows = _merge_into_windows(segments, composite)

    mean = statistics.mean(composite)
    stdev = statistics.pstdev(composite) if len(composite) > 1 else 0.0
    threshold = mean + THRESHOLD_STDEV_MULT * stdev

    qualifying = [w for w in windows if w["score"] > threshold]
    if not qualifying:
        raise NoHighlightFoundError(
            f"No segment scored above the highlight threshold ({threshold:.3f})."
        )

    # Explicit tie-break on start time keeps selection deterministic even
    # when two windows score exactly equal.
    qualifying.sort(key=lambda w: (-w["score"], w["start"]))

    # Defensive overlap guard: on real (time-ordered) Whisper output,
    # _merge_into_windows already can't produce overlapping windows, so
    # this is normally a no-op. It exists for robustness against a
    # transcript with out-of-order segments, which would otherwise be
    # able to produce overlapping windows here.
    selected: list[dict] = []
    for window in qualifying:
        if any(window["start"] < s["end"] and window["end"] > s["start"] for s in selected):
            continue
        selected.append(window)

    for window in selected:
        window["segment_scores"] = {
            sid: segment_scores_by_id[sid] for sid in window["segment_ids"]
        }

    logger.info(
        "Selected %d highlight window(s) from %d candidate(s) (threshold=%.3f).",
        len(selected),
        len(qualifying),
        threshold,
    )
    return selected


def _keyword_score(text: str) -> float:
    lowered = text.lower()
    # Alternation (not a merged character class) so an unspaced
    # Latin+Devanagari run can't fuse into one bogus token.
    words = re.findall(r"[a-z']+|[ऀ-ॿ]+", lowered)
    word_count = max(len(words), 1)
    keyword_hits = sum(1 for w in words if w in ALL_KEYWORDS)
    punctuation_hits = text.count("!") + text.count("?")
    return (keyword_hits / word_count) + (0.5 * punctuation_hits)


def _pause_scores(segments: list[dict]) -> list[float]:
    scores = []
    for i, seg in enumerate(segments):
        before = seg["start"] - segments[i - 1]["end"] if i > 0 else 0.0
        after = segments[i + 1]["start"] - seg["end"] if i < len(segments) - 1 else 0.0
        scores.append(max(before, 0.0) + max(after, 0.0))
    return scores


def _rate_score(segment: dict) -> float:
    duration = max(segment["end"] - segment["start"], 0.01)
    word_count = len(segment.get("words") or segment["text"].split())
    return word_count / duration


def _energy_scores(video_path: Path, segments: list[dict]) -> list[float]:
    """Average RMS audio energy per segment, decoded via Whisper's own
    ffmpeg-backed loader (already a dependency -- no new audio library)."""
    audio = whisper_audio.load_audio(str(video_path), sr=SAMPLE_RATE)
    frame_len = int(SAMPLE_RATE * ENERGY_FRAME_MS / 1000)

    scores = []
    for seg in segments:
        start_sample = max(int(seg["start"] * SAMPLE_RATE), 0)
        end_sample = min(int(seg["end"] * SAMPLE_RATE), len(audio))
        if end_sample <= start_sample:
            scores.append(0.0)
            continue
        clip = audio[start_sample:end_sample]
        n_frames = max(len(clip) // frame_len, 1)
        clip = clip[: n_frames * frame_len]
        frames = clip.reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        scores.append(float(np.mean(rms)))
    return scores


def _zscore(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return [0.0 for _ in values]
    return [(v - mean) / stdev for v in values]


def _merge_into_windows(segments: list[dict], composite_scores: list[float]) -> list[dict]:
    """Merge adjacent above-floor segments (gap <= MERGE_GAP_SEC) into
    windows; a window's score is the max of its member segments' scores,
    so one sharp peak isn't diluted by averaging over a longer run."""
    floor = statistics.median(composite_scores)
    windows: list[dict] = []
    current: dict | None = None

    for seg, score in zip(segments, composite_scores):
        if score <= floor:
            if current:
                windows.append(current)
                current = None
            continue

        if current and seg["start"] - current["end"] <= MERGE_GAP_SEC:
            current["end"] = seg["end"]
            current["segment_ids"].append(seg["id"])
            current["score"] = max(current["score"], score)
        else:
            if current:
                windows.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "score": score,
                "segment_ids": [seg["id"]],
            }

    if current:
        windows.append(current)

    return windows

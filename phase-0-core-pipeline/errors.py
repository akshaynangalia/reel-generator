"""
Custom exceptions raised by pipeline stages.

pipeline.py catches these centrally and exits cleanly with a clear log
message instead of letting a raw traceback surface to the user.
"""


class PipelineError(Exception):
    """Base class for expected, user-facing pipeline failures."""


class InputTooShortError(PipelineError):
    """Input video is shorter than the minimum clip length (30s)."""


class NoSpeechDetectedError(PipelineError):
    """Whisper found no speech in the input video's audio track."""


class NoHighlightFoundError(PipelineError):
    """No transcript segment scored high enough to qualify as a highlight."""

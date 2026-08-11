# CLAUDE.md — Phase 0: Core Pipeline

## What this phase is

A standalone, no-UI CLI script that proves out the full local AI
processing pipeline end-to-end: `pipeline.py <input_video>` transcribes
the video (Whisper), scores the transcript to find highlight-worthy
segments (deterministic heuristic, no paid model), cuts 30–60s clips,
reframes each to 9:16 with face tracking (MediaPipe, center-crop
fallback), burns in styled captions (ffmpeg/libass), and writes finished
`final.mp4` files to `output/<video_stem>/clip_NN_<start>-<end>s/`.

Run it with:
```
python pipeline.py <input_video> [--output-dir DIR] [--model-size base] [--max-clips 3] [--force-retranscribe]
```
Dependencies: `pip install -r requirements.txt` plus a system `ffmpeg` on
PATH (not pip-installable). Tested against Windows 11 with ffmpeg 9.0
(full build, `libass`/`libx264` present) and Python 3.12 (venv).

## Module map

- `errors.py` — `PipelineError` and its subclasses
  (`InputTooShortError`, `NoSpeechDetectedError`, `NoHighlightFoundError`),
  caught centrally in `pipeline.py` for clean exits.
- `transcribe.py` — Whisper `"base"` model, `fp16=False` (CPU-only),
  `word_timestamps=True`. Caches `transcript.json` by mtime comparison
  against the input video.
- `highlight_detection.py` — deterministic keyword / audio-energy /
  pause / speech-rate z-scored composite, merged into windows, top-N
  selected above `mean + 1.0*stdev`.
- `segment.py` — expands/clamps a window to 30–60s, snaps edges to word
  boundaries, cuts via ffmpeg (`-ss`/`-to` output seeking, video
  re-encoded, audio `-c:a copy`).
- `face_track.py` — MediaPipe Tasks API `FaceDetector` (VIDEO mode),
  samples every ~200ms, largest-bbox + most-central face selection, EMA
  smoothing, center-crop fallback if no face is ever detected.
- `reframe.py` — `sendcmd`-driven moving `crop` filter for the
  face-tracked path, static `crop` for the fallback; both scale to
  1080x1920; audio always `-c:a copy`.
- `captions.py` — chunks word timestamps into short lines, writes ASS,
  burns in via ffmpeg's `ass` filter.
- `pipeline.py` — CLI + orchestration + centralized error handling.

## Gotchas found during implementation

- **`mediapipe==1.0.0` (the version this project's venv has installed)
  has no `mediapipe.solutions` module at all** — `mp.solutions` raises
  `AttributeError`. Only the modern Tasks API
  (`mediapipe.tasks.python.vision.FaceDetector`) exists. This is a
  version-specific fact, not a general MediaPipe limitation — if the
  installed version ever changes, re-check this before assuming either
  API is available. The Tasks API needs a `.tflite` model asset, which is
  **not bundled** with the pip package; `face_track.py` downloads
  `blaze_face_short_range.tflite` from Google's public MediaPipe model
  storage on first use and caches it at
  `~/.cache/reel-generator/models/` (outside the repo, like Whisper's own
  weight cache, so no `.gitignore` entry is needed for it).

- **ffmpeg filtergraph strings can't safely contain raw Windows paths.**
  Both `ass=<path>` (captions) and `sendcmd=f=<path>` (reframe) break if
  given an absolute Windows path directly, because `:` inside the
  filtergraph string is ffmpeg's option separator (`C:\...` gets
  misparsed). Fix used throughout: run the ffmpeg subprocess with `cwd`
  set to the clip's own output directory and pass a bare filename.

- **`-c copy` is not safe for the initial highlight cut in `segment.py`**,
  even though the project's non-negotiable rule is "never touch the
  audio." Stream-copy only works there for *audio* (sample-accurate,
  not keyframe-constrained) — video stream-copy can only cut on
  keyframes, drifting the requested cut point by up to a GOP length. Since
  `reframe.py` re-encodes the video anyway (cropping changes pixels),
  there's no accuracy or performance reason to insist on video copy at
  the cut stage. Audio is `-c:a copy` at every ffmpeg stage; video is
  re-encoded at the cut and reframe stages, and again at the caption
  burn-in stage (ffmpeg filters require re-encoding the stream they
  touch).

- **Word-boundary snapping in `segment.py` can silently break the
  30–60s bound if done naively.** Each edge (start, end) is snapped
  independently to the nearest word boundary within a 1s tolerance; in
  one real test run this pulled *both* edges inward at once and produced
  a 29.56s clip (just under the 30s minimum). Fixed by re-checking the
  snapped length against `[MIN_CLIP_SEC, MAX_CLIP_SEC]` afterward and
  reverting whichever single edge is responsible (falling back to
  reverting both) rather than trusting the snap result unconditionally.

- **Whisper hallucinates on pure silence** rather than returning zero
  segments — observed directly on a synthetic silent test clip, which
  produced two spurious `"you"` segments. Only one of the two cleared the
  original `no_speech_prob > 0.9` check, so the "no speech" edge case
  wasn't reliably caught by that signal alone. Added a `MIN_WORD_COUNT`
  (5) check in `transcribe.py` as a second, more robust signal — a real
  transcript of any usable video has far more than 5 words, so a
  near-empty word count is a reliable silence indicator regardless of
  individual segment confidence scores.

- **Equal-weighted highlight scoring under-selects an "obvious" keyword
  highlight on flat/monotone audio.** In testing (TTS-narrated audio
  with genuinely flat prosody), a segment containing "amazing... 
  incredible... insane" scored highest of all segments on the keyword
  signal alone, but with all four signals weighted equally its composite
  score narrowly missed the selection threshold, diluted by ordinary
  pause/rate noise elsewhere in the transcript. Bumped `W_KEYWORD` from
  1.0 to 1.5 in `highlight_detection.py` — explicit exclamatory language
  is the most direct, least noisy signal of an intentional highlight, so
  it's now weighted somewhat above the others. Worth revisiting with more
  real (non-synthetic, non-monotone) sample videos as they become
  available.

- **`sendcmd`'s per-command hold is a step function, not interpolation.**
  The moving crop in `reframe.py` jumps between sampled crop positions
  (~5/sec) rather than smoothly interpolating between them. At the
  current sampling rate combined with `face_track.py`'s EMA smoothing
  this reads as acceptably smooth in testing, but true inter-sample
  interpolation is a reasonable future improvement, not attempted here.

## Testing performed

- A synthesized ~102s test video (Windows TTS narration over a static
  test pattern, no face) validated: transcription, highlight scoring
  and selection (including the keyword-weighting fix), 30–60s clip
  bounds (including the word-snap fix), the no-face center-crop
  fallback, caption burn-in and sync, audio preservation (`ffprobe`
  codec/rate/channel comparison against source), and determinism
  (identical clip selection across two full runs on the same input).
- A real ~4.6 minute news-broadcast clip (provided by the user, contains
  a visible face, Hindi audio) validated the face-tracked crop path
  end-to-end (`face_detected=True` on all 3 produced clips, valid 9:16
  output, captions burned in and synced, audio preserved).
- Edge cases: a 5s clip correctly raises `InputTooShortError` before
  transcription starts (fast fail); a synthetic silent clip correctly
  raises `NoSpeechDetectedError` after the `MIN_WORD_COUNT` fix. Both
  exit with a clear single log line and exit code 1, no traceback.
- Not yet tested: non-English content's effect on keyword scoring
  quality (the keyword list is English-only, so keyword-signal quality
  will be weaker on non-English audio — the real test video happened to
  be in Hindi, and Whisper romanized the captions rather than rendering
  Devanagari); very long (multi-hour) source videos; multiple simultaneous
  faces in frame.

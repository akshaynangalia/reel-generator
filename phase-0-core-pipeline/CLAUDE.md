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

## Phase 0 fixes pass (`.claude/specs/phase-0-fixes.md`)

Three defects found in the testing above were fixed, plus one additional
issue found only while verifying the fix (see below). Spec:
`.claude/specs/phase-0-fixes.md`.

### 1. Overlapping highlight clips

The overlap wasn't where the spec assumed. `_merge_into_windows` builds
windows by scanning time-ordered segments, so `detect_highlights`'s own
selection can't actually produce overlapping windows on real Whisper
output — a defensive guard was still added there (`highlight_detection.py`,
now checks each candidate against already-selected windows before
accepting), since it's cheap and matches what the spec's Definition of
Done tests, but on real transcripts it's a no-op. The **real** overlap
came from `segment.py`'s `build_clip_bounds`, which expands any
sub-30s window out to the 30-60s minimum *after* selection — two
originally non-overlapping windows can each grow into each other's
space. Fixed in `pipeline.py`: `detect_highlights` now returns the
**entire** qualifying candidate pool (no `max_clips` cap), and a new
`_select_non_overlapping_clips` computes each candidate's *expanded*
bounds and greedily accepts up to `max_clips` non-overlapping ones in
score order, falling back to lower-ranked candidates when higher-ranked
ones collide post-expansion.

**Gotcha, caught in plan review before implementation:** an earlier
draft of this fix still called `detect_highlights(..., max_clips=...)`,
capping the pool *before* the overlap filter ran — meaning a cluster of
top-scoring candidates that collide after expansion would silently
produce fewer than `max_clips` clips, with no lower-ranked candidates
left to fall back on. This is exactly what the real Hindi test video
does (see verification below): the top 3 scoring windows cluster and
two get rejected for overlap, so the pool has to be uncapped for the
3rd- and 4th-best candidates to be available as fallbacks.

Verified on the real ~4.6min Hindi video: `detect_highlights` returned
11 qualifying windows; `_select_non_overlapping_clips` rejected 2 for
post-expansion overlap and produced 3 genuinely non-overlapping final
clips. The same behavior (different rejected candidates) reproduced on
a second run with a different Whisper model size, confirming it's not
an artifact of one specific transcript.

### 2. Hindi audio: romanized transcription + English-only keywords

Two sub-fixes, plus one important correction to the spec's diagnosis
found during verification.

**Keyword scoring** (`highlight_detection.py`): added
`HIGHLIGHT_KEYWORDS_HI` (Devanagari), extended the tokenizer regex from
`[a-z']+` to `[a-z']+|[ऀ-ॿ]+` (alternation, not a merged character
class, so an unspaced Latin+Devanagari run can't fuse into one bogus
token), and `_keyword_score` now checks the **union** of both keyword
sets unconditionally on every segment — never switched per-video — so
mixed Hindi-English ("Hinglish") segments score hits from whichever
language a phrase actually uses. Verified directly (the real test
video's actual content didn't happen to contain any curated keyword
either way, so this needed a targeted check): Hindi-only, English-only,
and mixed-language test sentences all scored correctly, with the mixed
sentence confirming both lists are checked in the same call.

**Language detection** (`transcribe.py`): added automatic detection —
`_detect_language` runs Whisper's `detect_language` on a short (30s)
audio sample (via a new `_load_audio_sample`, a dedicated bounded
ffmpeg decode rather than reusing `whisper.audio.load_audio`, which has
no duration limit and would otherwise decode a long file twice), then
pins the result for the full `model.transcribe(..., language=...)`
call. An optional `language` param/`  --language` CLI flag can override
detection entirely.

**Important correction to the spec's root-cause diagnosis, found only
by actually re-running the real Hindi test video after implementing the
language-pinning fix as specified:** the transcript was *still*
romanized after the fix, identical to before it. Investigation found
the pre-fix cached transcript *already* had `"language": "hi"` —
Whisper's own internal auto-detect was already correctly identifying
Hindi. Explicitly pinning the same value Whisper would have auto-detected
anyway changes nothing about decoding, so language *detection* was never
the actual cause of the romanization. Empirically confirmed by re-running
the same video with `--model-size small`: it produced correct Devanagari
script, while `base` (the project's default) reliably romanizes Hindi
regardless of correct language identification — this is a known
characteristic of smaller Whisper models' decoders, not a language-ID
problem. Fix (added beyond the original spec's scope, after confirming
the above with the user): `transcribe_video`'s `model_size` now defaults
to `None` ("auto"), starting at `DEFAULT_MODEL_SIZE = "base"` for
detection and silently escalating to `ESCALATED_MODEL_SIZE = "small"`
before the full transcription **only** when the resolved language isn't
English and the caller didn't explicitly request a model size — English
transcription stays on the fast/cheap default; non-English content
auto-upgrades with no flag needed, matching the spec's original "just
work automatically" goal (which the language-pinning fix alone did not
actually achieve). The cached transcript's `model_size` field records
whichever model actually ran, not the initial one.

Verified end-to-end with zero flags (`python pipeline.py
sample_input/test_video.mp4`) on the real Hindi video: log shows
`Auto-detected language: hi` immediately followed by `escalating Whisper
model 'base' -> 'small'`, and the resulting `transcript.json` contains
correct Devanagari script throughout.

### 3. Face-tracking crop freeze on detection loss

All logic lives in `face_track.py`; `reframe.py` needed **no changes** —
`_write_sendcmd_file` already iterates the `centers` list blindly by
`{"t", "cx", "cy"}` shape, so inserting synthetic center-crop entries
with that same shape is invisible to it (a smaller diff than the spec's
illustrated `"held"`-flag/`timeout_ranges` schema change reaching
`reframe.py`, with identical ffmpeg-observable behavior).

New `FACE_TIMEOUT_SEC = 1.75` constant and `_split_into_runs`, which
splits `_detect_face_centers`'s already-gap-containing output into
contiguous runs wherever a gap exceeds the timeout. `track_face` then
smooths each run independently (a fresh `_smooth_centers` call per run
naturally resets the EMA, so tracking never eases back in from a stale
pre-gap position — it cuts cleanly instead) and inserts one synthetic
center-crop entry between runs at `prev_run_end_t + FACE_TIMEOUT_SEC`,
plus a trailing one if the face is lost near the clip's end and never
reacquired (using `cv2.CAP_PROP_FRAME_COUNT`, skipped gracefully if
unavailable/unreliable on a given container).

Verified with a synthetic occlusion scenario (unit-level, not a real
video — no sample clip with a suitable mid-clip occlusion was
available): a tracked run 0.0-2.0s, a 3s gap, then a tracked run
5.0-7.0s produces exactly `[tracked...2.0] -> [fallback@3.75] ->
[tracked from 5.0, fresh EMA]`, and a synthetic <1.5s gap correctly
does *not* split into a separate run.

### Also verified in this pass

- `-c:a copy` still honored end-to-end: `ffprobe` codec/rate/channel
  comparison of produced clips against source is an exact match.
- All modules byte-compile cleanly; the full pipeline runs clean
  (exit 0) on the real Hindi video with default arguments.

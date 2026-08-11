# Spec: Phase 0 Core Pipeline

## Overview

A local, no-UI Python script that proves out the full AI processing pipeline
described in CLAUDE.md's Product Pipeline before any FastAPI or Next.js code
is written. Given a path to a raw video file, the script transcribes it with
local Whisper, scores the transcript to find highlight-worthy segments,
cuts those segments into 30–60s clips, reframes each clip to 9:16 vertical
using MediaPipe/OpenCV face tracking (center-crop fallback if no face is
found), burns in styled captions with ffmpeg, and writes the finished MP4s
to a local output folder. This is steps 2–6 and 9 of the Product Pipeline
(transcription, highlight detection, segmentation, vertical reframe,
caption styling, MP4 output) with steps 1, 7, and 8 (upload, gallery,
selection) explicitly deferred to later phases since they require UI. No
adjustments were needed to the brief — it matches the root CLAUDE.md's
description of this phase almost exactly.

## Depends on

None — this is the foundational phase. It establishes the pipeline that
Phase 1 (web upload) and later phases will wrap with an API and UI.

## Pipeline / processing changes

This phase *is* the pipeline. All logic is new:

- `phase-0-core-pipeline/transcribe.py` — runs local Whisper on the input
  video's audio track, produces a timestamped transcript (word- or
  segment-level timing) saved as JSON. Pinned to the Whisper "base" model
  size — hardware is CPU-only (no GPU), and "base" is the right tradeoff
  between transcription quality and CPU inference time.
- `phase-0-core-pipeline/highlight_detection.py` — scores transcript
  segments using a keyword/energy/pause-based heuristic (no paid AI model,
  per CLAUDE.md) and selects candidate highlight windows.
- `phase-0-core-pipeline/segment.py` — expands/clamps each highlight window
  to a 30–60s clip boundary and cuts it from the source video via ffmpeg.
- `phase-0-core-pipeline/face_track.py` — uses MediaPipe/OpenCV to detect
  and track the speaking face per clip, producing a per-frame or
  per-interval crop center; falls back to a fixed center-crop if no face is
  detected anywhere in the clip.
- `phase-0-core-pipeline/reframe.py` — applies the crop centers from
  `face_track.py` via ffmpeg to produce a 9:16 vertical version of each
  clip.
- `phase-0-core-pipeline/captions.py` — converts the relevant transcript
  slice for each clip into a styled subtitle file and burns it into the
  vertical clip via ffmpeg.
- `phase-0-core-pipeline/pipeline.py` — orchestrates the above in order for
  a single input video and writes final MP4s to an output folder.

## Backend changes (FastAPI)

No backend changes. This phase is a standalone script/CLI, run manually
against a sample video. FastAPI wiring is explicitly out of scope until
Phase 1.

## Frontend changes (Next.js)

No frontend changes.

## Files to change

- `.gitignore` (root) — add `phase-0-core-pipeline/sample_input/` and
  `phase-0-core-pipeline/output/` now, so these local-only video/output
  folders are never committed from the start of this phase.

## Files to create

- `phase-0-core-pipeline/pipeline.py` — CLI entry point / orchestrator
- `phase-0-core-pipeline/transcribe.py`
- `phase-0-core-pipeline/highlight_detection.py`
- `phase-0-core-pipeline/segment.py`
- `phase-0-core-pipeline/face_track.py`
- `phase-0-core-pipeline/reframe.py`
- `phase-0-core-pipeline/captions.py`
- `phase-0-core-pipeline/requirements.txt` — pinned versions of this
  phase's dependencies
- `phase-0-core-pipeline/sample_input/` — placeholder folder for a test
  video (gitignored; not committed)
- `phase-0-core-pipeline/output/` — placeholder folder for generated clips
  (gitignored; not committed)
- `phase-0-core-pipeline/CLAUDE.md` — written at the end of the phase,
  documenting what was actually built and any gotchas found

## New dependencies

All confirmed already present in the project's local `venv` (already
installed, zero-cost, no API keys required):

- `openai-whisper` (+ `torch`) — local speech-to-text
- `opencv-python` (`cv2`) — video frame handling / crop application
- `mediapipe` — face detection for speaker tracking
- `ffmpeg` — confirmed available on PATH (system binary, not a Python
  package), used for cutting, cropping, and caption burn-in

No paid services, no API keys, nothing requiring user approval.

## Rules for implementation

- No paid APIs or services — every AI/processing component must run
  locally (Whisper, OpenCV/MediaPipe, ffmpeg) unless the user has
  explicitly approved an exception for this feature
- Original audio must never be altered, dubbed, or replaced — reframe and
  caption steps operate on video/subtitle streams only and must copy the
  original audio stream through unmodified
- This phase must be testable end-to-end on its own before the next phase
  begins — running `pipeline.py` against one sample video must produce
  finished vertical, captioned MP4 clips with no FastAPI/Next.js
  involvement
- Do not add infrastructure (databases beyond SQLite, cloud storage, auth)
  ahead of when it's actually needed — this phase needs none of that; all
  I/O is local disk
- Follow the existing folder convention: this phase's code lives entirely
  in `/phase-0-core-pipeline/`
- Prefer clear, well-commented code over cleverness, per CLAUDE.md's
  coding standards — this is the first phase and sets the tone for
  readability going forward
- Large model downloads (Whisper weights) and sample/output video files
  must not be committed to git — extend `.gitignore` as needed
- Handle these edge cases gracefully — log a clear, human-readable message
  and exit cleanly (non-zero exit code where appropriate) rather than
  crashing or failing silently:
  - Input video is shorter than 30s (too short to produce a valid clip)
  - Whisper detects no speech in the audio track
  - No transcript segment scores high enough to qualify as a highlight

## Definition of done

- [ ] `python phase-0-core-pipeline/pipeline.py <input_video>` runs
      start-to-finish on a real sample video without manual intervention
- [ ] A timestamped transcript is produced and is inspectable as JSON
- [ ] At least one highlight segment is correctly identified from a sample
      video that has an obvious highlight (e.g. a laugh, a strong claim, an
      energetic moment)
- [ ] Each selected highlight is cut into a clip between 30–60s long
- [ ] Each clip is reframed to 9:16 vertical; on a clip with a visible
      speaking face, the crop keeps the face roughly centered across the
      clip; on a clip with no detectable face, the center-crop fallback
      produces a valid 9:16 output instead of erroring
- [ ] Each output clip has burned-in captions that are readable and
      reasonably synced to the spoken audio
- [ ] The original audio track in every output clip is carried through via
      ffmpeg's `-c:a copy` (stream copy, no re-encoding, no dubbing, no
      synthetic voice)
- [ ] Running the pipeline twice on the same input produces consistent
      (not wildly different) highlight selections and crops
- [ ] `phase-0-core-pipeline/CLAUDE.md` is written documenting what was
      actually built and any gotchas found

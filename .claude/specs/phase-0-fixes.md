Spec: Phase 0 Fixes

## Overview

Three defects were found while manually testing Phase 0's core pipeline
against a real sample video (a Hindi-language news broadcast) and are
fixed here without changing the pipeline's overall shape: (1)
`highlight_detection.py`'s top-N window selection can select multiple
windows that heavily overlap in time, because qualifying windows are
sorted and sliced by score alone with no overlap check — this must be
rejected during selection so `--max-clips` windows are always
non-overlapping. (2) `HIGHLIGHT_KEYWORDS` in `highlight_detection.py` is
English-only, and `transcribe.py` lets Whisper auto-detect language,
which on the Hindi test video produced romanized/garbled text instead of
Devanagari — a Hindi keyword list must be added alongside the English
one, checked together with the English list on every segment (not
switched between per-video, since real content is often mixed
Hindi-English "Hinglish"), and the pipeline must automatically detect the
spoken language from a short sample of the audio and pin that detected
language for the full transcription, rather than relying on Whisper's
own full-clip auto-detect, which is the path that produced the garbled
romanized text on the Hindi test video. (3) `face_track.py`'s
`_detect_face_centers` only appends a center when a face is found, so a
detection gap mid-clip (occlusion, face turning away, etc.) leaves no
timestamps to smooth or hold in `reframe.py`'s `sendcmd` filter, which
holds the last known crop position as a step function until the next
sample — this reads as the crop "freezing" on a stale position for the
whole gap; a timeout must force a fallback to a safe center crop once a
detection gap exceeds ~1.5–2s. This is grounded in
`phase-0-core-pipeline/CLAUDE.md`'s documented gotchas (the
`sendcmd` step-function behavior and the "not yet tested" note on
non-English keyword scoring) and in the actual source of
`highlight_detection.py`, `transcribe.py`, `face_track.py`, and
`reframe.py`. Nothing in the brief needed adjustment against
`CLAUDE.md`'s constraints — all three fixes are local, zero-cost, and
stay within Phase 0's existing CLI-script scope.

## Depends on

Phase 0's existing pipeline modules and their current behavior:
`highlight_detection.py`'s window-scoring/selection, `transcribe.py`'s
Whisper invocation and transcript cache format, and
`face_track.py`/`reframe.py`'s crop-center/`sendcmd` contract.

## Pipeline / processing changes

- **`highlight_detection.py` — overlap rejection in `detect_highlights`.**
  After sorting `qualifying` windows by `(-score, start)` (existing tie-
  break, kept), select into `selected` one at a time and skip any
  candidate window that overlaps (`start < already_selected.end and end >
  already_selected.start` for any already-picked window) with a window
  already selected, continuing down the sorted list until `max_clips` are
  picked or candidates are exhausted. This preserves the current
  highest-score-first priority while guaranteeing no time-overlap between
  the windows a single pipeline run outputs. No change to
  `_merge_into_windows` or the scoring signals themselves.

- **`highlight_detection.py` — Hindi keyword list, checked unconditionally.**
  Add a `HIGHLIGHT_KEYWORDS_HI` set of common Hindi (Devanagari-script)
  emphatic/highlight words, sitting alongside the existing
  `HIGHLIGHT_KEYWORDS` (kept as the English list). `_keyword_score` must
  check **both** lists against every segment's text regardless of the
  transcript's detected language — it must not branch or switch lists
  based on a single whole-video language value, since real content
  (especially the Hindi test video's likely register) can mix Hindi and
  English within the same segment ("Hinglish"), and a per-video language
  switch would miss English keyword hits in an otherwise-Hindi segment
  or vice versa. The current `re.findall(r"[a-z']+", ...)` word-
  tokenization only matches Latin script and would silently score zero
  Hindi keyword hits even with a Hindi list present, so the tokenizer
  must be extended to also capture Devanagari word characters (Unicode
  range `ऀ-ॿ`), and `_keyword_score` must count hits against the union of
  both keyword sets on the combined token list.

- **`transcribe.py` — automatic language detection, then pinned
  transcription.** Replace the current single `model.transcribe(...)`
  call (which lets Whisper auto-detect language internally over the
  whole clip — the unreliable path that produced romanized/garbled text
  on the Hindi test video) with two steps: first, run Whisper's
  language-detection on a short audio sample (a few seconds, e.g. via
  `whisper.audio.load_audio`/`pad_or_trim` + `model.detect_language`, the
  standard Whisper pattern for this) to get a detected language code;
  second, call `model.transcribe(..., language=detected_language)` with
  that code explicitly pinned for the full transcription. This requires
  no user action — Hindi and English input are both handled automatically
  with no flag needed. `transcribe_video` still accepts an optional
  `language: str | None = None` override parameter for edge cases (e.g.
  detection guessing wrong on unusual audio); when explicitly provided,
  it skips the detection step and pins straight to the given language.
  `pipeline.py` gains an optional `--language` CLI flag (default `None`)
  wired through to this override parameter, but it is never required for
  correct behavior on either English or Hindi input. The cached
  `transcript.json`'s existing `"language"` field is unchanged in shape;
  it now reflects the detected (or override) language that was actually
  pinned for transcription.

- **`face_track.py` — detection-gap timeout.** In `_detect_face_centers`,
  track the timestamp of the last successful detection. When a sampled
  frame's timestamp exceeds `last_detected_t + FACE_TIMEOUT_SEC` (new
  constant, default 1.75s — within the requested 1.5–2s range) without an
  intervening detection, mark that gap so the caller can react, rather
  than silently leaving a hole in `centers` for `_smooth_centers`/
  `reframe.py` to paper over with a held stale position. Concretely: the
  centers list gains an explicit `"held": bool` (or equivalent) marker,
  or gaps are surfaced as a separate `timeout_ranges: [{"start": t, "end":
  t}]` list on the dict `track_face` returns — either approach must reach
  `reframe.py`. `reframe.py`'s `_write_sendcmd_file` must then, for any
  gap so marked, insert `sendcmd` entries that explicitly move the crop
  back to the safe center-crop position (from `_clamp_center(width/2,
  height/2, ...)`, the same fallback already used for the no-face-at-all
  case) at gap start, and only resume face-tracked positions once
  detection resumes — instead of leaving the last pre-gap `sendcmd`
  command in effect for the gap's full duration. `EMA_ALPHA` smoothing
  must not smooth across a timed-out gap (i.e. the EMA resets when
  tracking resumes after a timeout, rather than easing in from the
  stale pre-gap position, which would reintroduce a slow drift back onto
  the face instead of a clean cut to center-crop and back).

## Backend changes (FastAPI)

No backend changes — Phase 0 is still the standalone CLI script; no
FastAPI app exists yet.

## Frontend changes (Next.js)

No frontend changes — no UI exists yet in Phase 0.

## Files to change

- `phase-0-core-pipeline/highlight_detection.py`
- `phase-0-core-pipeline/transcribe.py`
- `phase-0-core-pipeline/face_track.py`
- `phase-0-core-pipeline/reframe.py`
- `phase-0-core-pipeline/pipeline.py` (new optional `--language` CLI
  override flag, passed through to `transcribe.transcribe_video`; not
  required for normal use)
- `phase-0-core-pipeline/CLAUDE.md` (append what was actually fixed and
  any new gotchas found, per the project's documented process — do not
  rewrite the existing gotchas already recorded there)

## Files to create

No new files — all three fixes land inside Phase 0's existing modules.

## New dependencies

No new dependencies. All three fixes use only what's already installed
(Python stdlib `re`/Unicode handling, existing Whisper/MediaPipe/ffmpeg
usage).

## Rules for implementation

- No paid APIs or services — every AI/processing component must run
  locally (Whisper, OpenCV/MediaPipe, ffmpeg) unless the user has
  explicitly approved an exception for this feature
- Original audio must never be altered, dubbed, or replaced
- This phase must be testable end-to-end on its own before the next phase
  begins
- Do not add infrastructure (databases beyond SQLite, cloud storage, auth)
  ahead of when it's actually needed for this phase
- Follow the existing folder convention: these are fixes to
  `phase-0-core-pipeline/`, not a new phase folder
- Language handling must be fully automatic by default: with `--language`
  omitted, English input must still transcribe correctly via the new
  detect-then-pin flow (detection should correctly identify English and
  pin it, producing equivalent output to today's auto-detect on the
  existing English test video), and the overlap-rejection change must not
  change output on inputs that already produced non-overlapping windows
  (verify against the existing ~102s synthetic test video)
- Keep the fix scoped to these three defects — do not use this pass to
  re-tune unrelated scoring weights, sampling rates, or thresholds
  documented in the existing `CLAUDE.md` gotchas

## Definition of done

- [ ] On an input where two or more qualifying highlight windows overlap
      in time, `detect_highlights` returns only non-overlapping windows
      (verified with a constructed transcript/test case that forces an
      overlap, since the existing sample videos may not naturally
      produce one)
- [ ] Re-running the pipeline against the existing ~102s synthetic
      English test video still selects the same clips as before this
      change (confirms the overlap fix is a no-op when no overlap exists)
- [ ] Running `pipeline.py` with no `--language` flag against the real
      Hindi test video automatically detects Hindi from the short audio
      sample, pins it for the full transcription, and produces a
      `transcript.json` with Devanagari text (not romanized/garbled Latin
      transcription) — no user-supplied flag involved
- [ ] Running `pipeline.py` with no `--language` flag against the
      existing English test video automatically detects English and
      transcribes correctly, with output equivalent to today's behavior
- [ ] The optional `--language` override, when explicitly passed, skips
      detection and pins straight to the given language (spot-checked
      against one input)
- [ ] With a Hindi (or Hinglish) transcript, at least one highlight
      window's keyword score reflects a real Devanagari keyword hit, and
      a mixed Hindi-English segment scores hits from both keyword lists
      rather than only one (spot-checked via `segment_scores`/logging,
      not just a non-zero composite score)
- [ ] A constructed/synthetic test clip with a face detected, then
      occluded for >2s mid-clip, then visible again produces a
      `final.mp4` whose crop visibly (frame-inspectable via `ffprobe`/
      extracted frames) falls back to center-crop during the occlusion
      instead of holding the pre-occlusion crop position for the whole
      gap
- [ ] A short (<1.5s) detection gap does *not* trigger the fallback (EMA-
      smoothed tracking continues through brief, normal single-frame
      misses) — confirms the timeout threshold isn't so aggressive it
      defeats normal tracking robustness
- [ ] Audio in all produced clips has `-c:a copy` honored, no
      re-encoding, no dubbing (`ffprobe` codec/rate/channel comparison
      against source, per the existing Phase 0 testing convention)
- [ ] `phase-0-core-pipeline/CLAUDE.md` updated with what was actually
      fixed and any new gotchas found during implementation

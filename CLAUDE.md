# CLAUDE.md — AI Reel Generator

## Project Overview

An AI-powered web application that takes a long-form video and automatically
generates short, caption-ready vertical clips (Instagram Reels / YouTube
Shorts style). The user uploads a video, the system transcribes it, detects
highlight-worthy segments, crops to vertical format while tracking the
speaking face, burns in styled captions, and presents a gallery of generated
reels for the user to preview, select, and download.

Original audio must always be preserved — no dubbing, no synthetic voice.

## Current Constraints (v1)

- **Video input:** direct file upload only. No YouTube/cloud link import yet
  — that is a planned future phase, not part of the current build.
- **Zero-cost requirement:** no paid APIs, no paid cloud services. Every
  component in the AI pipeline must run locally on open-source tools.
- **No auth yet:** authentication/multi-user support is a later phase.
  Early phases should assume a single local user.

## Tech Stack

- **Backend:** Python + FastAPI
- **Frontend:** Next.js (React)
- **Database:** SQLite (local file-based; acceptable until real multi-user
  traffic requires Postgres — do not add Postgres prematurely)
- **Speech-to-text:** local Whisper (open-source, runs on CPU, no API key)
- **Face detection / speaking-head tracking:** OpenCV + MediaPipe
- **Video processing (cutting, cropping, caption burn-in):** ffmpeg
- **Storage:** local disk during development. Cloud object storage
  (e.g. Cloudflare R2 free tier) only if/when we move toward deployment —
  do not add this prematurely.

Do not introduce paid services, additional cloud infrastructure, or
alternate frameworks without this being an explicit decision discussed and
recorded here.

## Product Pipeline (what the system must eventually do end-to-end)

1. User uploads a raw video file (large file support, progress indicator)
2. Speech-to-text transcription of the full video (Whisper)
3. Highlight detection — identify strong segments from the transcript
   (keyword/energy/pause-based scoring; no paid AI model)
4. Reel segmentation — cut into multiple 30–60s clips based on highlights
5. Vertical reframe — crop to 9:16 using face tracking so the speaker stays
   centered (fallback to center-crop if no face is detected)
6. Caption styling — burn in branded, readable captions per clip
7. Present a gallery of generated reels for preview
8. User selects the best clips
9. Download as MP4 (social auto-share is a later phase, not v1 — requires
   platform app review and is out of scope until the core pipeline works)

## Development Process — Spec-Driven, Phase by Phase

This project is built in phases, each covering one meaningful feature slice
(not just backend or just frontend — a phase includes whatever slice of UI
is needed to test that feature end-to-end). For each phase, the process is:

1. Run the `/spec` custom command to generate a spec file for that phase in
   `/specs/` — goals, scope, non-goals, file changes, acceptance criteria.
2. Review the spec before proceeding.
3. Enter Plan Mode referencing the approved spec to get an implementation
   plan (files to add/change, order of work, libraries needed).
4. Review the plan before any code is written.
5. Implement.
6. Test manually against a real sample video/output before moving on.
7. Commit with a clear message.
8. Write a `CLAUDE.md` inside that phase's folder documenting what was
   actually built (not the plan — the result), plus any gotchas found.

Do not skip the spec or plan-mode step, even for small changes — this is a
deliberate process choice for this project, not a suggestion.

## Folder Structure Conventions

- Each phase gets its own top-level folder, e.g. `/phase-0-core-pipeline`,
  `/phase-1-web-upload`, with its own `CLAUDE.md` once complete.
- Specs live in `/specs/`, one file per phase/feature.
- Custom Claude Code commands live in `/.claude/commands/`.
- This root `CLAUDE.md` covers project-wide, rarely-changing information
  only (stack, conventions, vision, constraints). It should be updated only
  when a project-wide decision changes — not incrementally per feature.
  Feature-specific detail belongs in that phase's own `CLAUDE.md`, not here.

## Coding Standards

- Prefer clear, well-commented code over cleverness — this project is being
  built by someone learning the tooling, and code should be readable on
  review, not just functional.
- No paid dependencies without explicit discussion first (see zero-cost
  constraint above).
- Every phase should be independently testable before the next phase
  begins.

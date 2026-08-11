---
description: Create a spec file for a new phase/feature in the AI Reel Generator
argument-hint: feature-slug: 2-line description of what to build and why, e.g. phase-0-core-pipeline: Local script that takes a video file and outputs captioned vertical clips. Proves the processing pipeline works before any UI is built.
allowed-tools: Read, Write, Glob
---

You are a senior developer scoping a new phase/feature for the AI Reel
Generator — a Python/FastAPI + Next.js app that converts long-form video
into caption-ready vertical reels. Always follow the rules in CLAUDE.md.

User input: $ARGUMENTS

## Step 1 — Parse the arguments

From $ARGUMENTS extract:
- **feature_title** — human readable title in Title Case (e.g. "Phase 0 Core
  Pipeline", "Reel Gallery UI")
- **feature_slug** — file-safe slug: lowercase, kebab-case, only a-z, 0-9 and
  -, maximum 40 characters (e.g. `phase-0-core-pipeline`, `reel-gallery-ui`)
- **feature_brief** — everything after the colon: the user's own 1–2 line
  description of what the feature should do and why

If you cannot infer these from $ARGUMENTS, ask the user to clarify before
proceeding.

## Step 2 — Check for duplicate specs

Read every file in `/specs/`. If a spec already exists for the same or a
very similar feature/phase, tell the user and ask whether to proceed
anyway, overwrite, or stop.

## Step 3 — Research the project

Read these before writing anything, if they exist:
- `CLAUDE.md` (root) — tech stack, constraints, product pipeline,
  conventions
- Any existing `/phase-*/CLAUDE.md` files — what's already been built, so
  this spec doesn't duplicate or conflict with prior phases
- Any existing `/specs/*.md` — prior specs, for consistency of style and to
  catch overlap

Early in the project these may not exist yet or may be sparse — that is
expected. In that case, rely on the root `CLAUDE.md` as the source of
truth and do not invent codebase details that haven't been built yet.

Compare feature_brief against what you find. If the brief conflicts with
the zero-cost constraint, the current v1 scope (upload-only, no auth), or
a decision already recorded in a phase's `CLAUDE.md`, flag this to the
user explicitly and ask how they want to proceed. Do not silently
reinterpret the brief to make it fit.

## Step 4 — Write the spec

Generate a spec document with this exact structure:

Spec: <feature_title>

## Overview

Expand the user's brief ("<feature_brief>") into one paragraph, grounded in
what Step 3 found (or, for early phases, grounded in the root CLAUDE.md).
Note explicitly if anything in the brief had to be adjusted, and why.

## Depends on

Which existing phases, modules, or behaviors this feature builds on (e.g.
"Phase 0's transcription output format", "the existing upload endpoint").
If this is the first phase: "None — this is the foundational phase."

## Pipeline / processing changes

Any new or modified logic in the video/audio processing pipeline
(transcription, highlight detection, segmentation, face tracking, caption
rendering). Reference exact file names where known. If none: "No pipeline
changes."

## Backend changes (FastAPI)

Any new or modified API routes, request/response shapes, or job-handling
logic. If none: "No backend changes."

## Frontend changes (Next.js)

Any new pages, components, or UI states (including loading/progress states
where relevant). If none: "No frontend changes."

## Files to change

Every existing file that will be modified.

## Files to create

Every new file that will be created, including the phase's own
`CLAUDE.md` if this phase doesn't have one yet.

## New dependencies

Any new packages needed. Flag clearly if a proposed dependency is a paid
service or requires an API key — this must be raised to the user before
proceeding, since the project is zero-cost by constraint. If none: "No new
dependencies."

## Rules for implementation

Always include, plus anything specific to this feature:

- No paid APIs or services — every AI/processing component must run
  locally (Whisper, OpenCV/MediaPipe, ffmpeg) unless the user has
  explicitly approved an exception for this feature
- Original audio must never be altered, dubbed, or replaced
- This phase must be testable end-to-end on its own before the next phase
  begins
- Do not add infrastructure (databases beyond SQLite, cloud storage, auth)
  ahead of when it's actually needed for this phase
- Follow the existing folder convention: this phase's code lives in its
  own `/phase-<n>-<name>/` folder

## Definition of done

A specific, testable checklist. Each item must be verifiable by running
the phase manually against a real sample video/output, and by any
automated tests this phase includes.

## Step 5 — Save the spec

Save to: `/specs/<feature_slug>.md`

(Create the `/specs/` folder first if it doesn't exist.)

## Step 6 — Report to the user

Print a short summary in this exact format:

Spec file: /specs/<feature_slug>.md
Title: <feature_title>

Then tell the user: "Review the spec at /specs/<feature_slug>.md, then
enter Plan Mode with Shift+Tab twice to begin implementation."

Do not print the full spec in chat unless explicitly asked.
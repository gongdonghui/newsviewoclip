# Short Broadcaster-Focused Clip Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change clip planning so the default output favors short, concentrated, broadcaster-led clips in the 15-45 second range.

**Architecture:** Tighten the DeepSeek prompt in `scripts/analyze_clips.py`, tighten fallback duration defaults and summary text in `scripts/run_pipeline.py`, and adjust fallback grouping in `scripts/build_clip_plan.py` so the local non-LLM path also favors shorter payoff windows. Keep the output schema, `mlx_whisper` transcription flow, and output isolation/concurrency logic unchanged.

**Tech Stack:** Python 3.12, `pytest`, `openai` client for DeepSeek, `mlx_whisper`, `ffmpeg`, `ffprobe`

---

## File Map

- Modify: `scripts/analyze_clips.py`
  - Tighten the LLM segmentation prompt toward 15-45 second broadcaster-first clips.
- Modify: `scripts/build_clip_plan.py`
  - Tighten fallback grouping so clips flush earlier and stay concentrated.
- Modify: `scripts/run_pipeline.py`
  - Update fallback duration defaults and printed strategy summary.
- Modify: `tests/test_run_pipeline.py`
  - Update fallback expectations for shorter durations.
- Create or modify: `tests/test_analyze_clips.py`
  - Add prompt-level tests for the new broadcaster-focused guidance.

### Task 1: Lock the New Analysis Prompt in Tests

**Files:**
- Create or Modify: `tests/test_analyze_clips.py`
- Modify: `scripts/analyze_clips.py`

- [ ] **Step 1: Write the failing test for prompt guidance**

```python
from scripts.analyze_clips import build_analysis_messages


def test_build_analysis_messages_requests_short_broadcaster_focused_clips():
    messages = build_analysis_messages(
        {"duration_seconds": 120.0},
        [{"id": 1, "start": 0.0, "end": 5.0, "text": "Anchor intro"}],
    )

    system_message = messages[0]["content"]
    assert "15-45" in system_message
    assert "broadcaster" in system_message.lower()
    assert "anchor" in system_message.lower()
    assert "brief reporter or soundbite continuation" in system_message.lower()
    assert "more clips that are tighter" in system_message.lower()
```

- [ ] **Step 2: Run the prompt test to verify it fails**

Run: `python3 -m pytest tests/test_analyze_clips.py -k broadcaster_focused -q`

Expected: FAIL because the current system prompt still describes generic semantically complete clips.

- [ ] **Step 3: Update the DeepSeek system prompt**

Implement the smallest change in `scripts/analyze_clips.py`:

- replace the generic “semantically complete clips” wording
- explicitly require 15-45 second clips
- explicitly prefer broadcaster-led clips
- allow only brief continuation when it sharpens the point
- tell the model to prefer tighter/more numerous clips over long packages

- [ ] **Step 4: Re-run the prompt test**

Run: `python3 -m pytest tests/test_analyze_clips.py -k broadcaster_focused -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_analyze_clips.py scripts/analyze_clips.py
git commit -m "test: lock broadcaster-focused clip prompt"
```

### Task 2: Tighten Fallback Duration Defaults

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `tests/test_run_pipeline.py`

- [ ] **Step 1: Write the failing test for shorter fallback windows**

Add or update a test around `build_plan_preview(...)`:

```python
def test_fixture_transcript_produces_shorter_clip_plan():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 12.0, "text": "Anchor line one."},
            {"id": 2, "start": 12.5, "end": 26.0, "text": "Anchor payoff."},
            {"id": 3, "start": 40.0, "end": 55.0, "text": "Next topic."},
        ]
    }

    clips = build_plan_preview(
        transcript,
        duration_seconds=120.0,
        min_clip_seconds=15,
        target_clip_seconds=30,
        max_clip_seconds=45,
        padding_before=0.5,
        padding_after=1.0,
    )

    assert clips[0]["duration_seconds"] <= 45.0
```

- [ ] **Step 2: Run the focused run-pipeline test**

Run: `python3 -m pytest tests/test_run_pipeline.py -k shorter_clip_plan -q`

Expected: FAIL if the current assertions or constants still reflect longer windows.

- [ ] **Step 3: Update fallback constants and printed strategy text**

Change `scripts/run_pipeline.py`:

- `MIN_CLIP_SECONDS = 15`
- `TARGET_CLIP_SECONDS = 30`
- `MAX_CLIP_SECONDS = 45`
- update printed strategy summary from `20-180 秒（目标 60-120 秒）`
- to text matching `15-45 秒` and broadcaster-focused behavior

- [ ] **Step 4: Re-run the focused run-pipeline test**

Run: `python3 -m pytest tests/test_run_pipeline.py -k shorter_clip_plan -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_run_pipeline.py scripts/run_pipeline.py
git commit -m "feat: tighten fallback clip duration defaults"
```

### Task 3: Tighten Fallback Grouping

**Files:**
- Modify: `scripts/build_clip_plan.py`
- Modify: `tests/test_run_pipeline.py`

- [ ] **Step 1: Write the failing test for earlier flush behavior**

Add a focused fallback-grouping test:

```python
from scripts.build_clip_plan import build_fallback_clips


def test_build_fallback_clips_prefers_short_payoff_windows():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 10.0, "text": "Anchor setup."},
            {"id": 2, "start": 10.2, "end": 24.0, "text": "Anchor payoff."},
            {"id": 3, "start": 24.2, "end": 40.0, "text": "Reporter background."},
        ]
    }

    clips = build_fallback_clips(
        transcript,
        min_clip_seconds=15,
        target_clip_seconds=30,
        max_clip_seconds=45,
    )

    assert len(clips) >= 1
    assert float(clips[0]["end"]) - float(clips[0]["start"]) <= 45
```

- [ ] **Step 2: Run the focused fallback test**

Run: `python3 -m pytest tests/test_run_pipeline.py -k fallback_clips -q`

Expected: FAIL if the fallback still groups into broader narrative bundles.

- [ ] **Step 3: Update fallback grouping minimally**

In `scripts/build_clip_plan.py`, make the smallest change that:

- flushes once a short concentrated block reaches the target window
- avoids carrying additional background segments into the same clip when already within a valid range
- preserves existing shape and schema

- [ ] **Step 4: Re-run the focused fallback test**

Run: `python3 -m pytest tests/test_run_pipeline.py -k fallback_clips -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_run_pipeline.py scripts/build_clip_plan.py
git commit -m "feat: tighten fallback clip grouping"
```

### Task 4: Run Regression Tests

**Files:**
- No new file changes expected unless regressions appear

- [ ] **Step 1: Run targeted regression suite**

Run:

```bash
python3 -m pytest \
  tests/test_analyze_clips.py \
  tests/test_probe_audio.py \
  tests/test_transcribe.py \
  tests/test_cut_clips.py \
  tests/test_export_clip_subtitles.py \
  tests/test_run_pipeline.py \
  -q
```

Expected: All tests pass.

- [ ] **Step 2: Fix regressions minimally if needed**

Only touch files directly implicated by failures.

- [ ] **Step 3: Re-run the regression suite**

Run the same command again.

Expected: All tests pass cleanly.

- [ ] **Step 4: Commit**

```bash
git add scripts tests
git commit -m "test: verify short broadcaster-focused clip strategy"
```

### Task 5: Validate on `l2.mp4` in Plan-Only Mode

**Files:**
- Generate: `output/l2-66136969/metadata/clip_plan.md`
- Generate: `output/l2-66136969/metadata/clip_plan.json`

- [ ] **Step 1: Remove or replace stale `l2` planning artifacts if needed**

Run:

```bash
rm -rf output/l2-66136969 temp/l2-66136969
```

Only do this if the user is comfortable replacing prior generated artifacts for the same input.

- [ ] **Step 2: Run the pipeline in plan-only mode**

Run:

```bash
python3 scripts/run_pipeline.py \
  --input ~/Downloads/l2.mp4 \
  --mode news \
  --language en \
  --plan-only \
  --force
```

Expected: Generates fresh transcript and a new clip plan in the isolated `l2` output directory.

- [ ] **Step 3: Review the plan output**

Inspect:

```bash
sed -n '1,200p' output/l2-66136969/metadata/clip_plan.md
```

Expected:

- more clips than before
- shorter durations
- tighter titles/summaries
- more anchor-led openings

- [ ] **Step 4: Record findings**

Summarize:

- clip count
- rough duration distribution
- whether broadcaster focus improved materially

- [ ] **Step 5: Commit code only, not generated outputs, unless explicitly requested**

```bash
git add scripts tests docs/superpowers/plans/2026-06-02-short-broadcaster-focused-clip-strategy.md
git commit -m "docs: add short broadcaster-focused clip strategy plan"
```

### Task 6: Optional Execute Validation

**Files:**
- Generate: `output/l2-66136969/clips/*.mp4`
- Generate: `output/l2-66136969/subtitles/*.srt`
- Generate: `output/l2-66136969/subtitles/*.vtt`

- [ ] **Step 1: Only proceed if the plan-only output looks better**

Decision gate: do not run execute if the revised plan is not materially improved.

- [ ] **Step 2: Run execute mode**

Run:

```bash
python3 scripts/run_pipeline.py \
  --input ~/Downloads/l2.mp4 \
  --mode news \
  --language en \
  --execute \
  --force
```

Expected: clip files and per-clip subtitles are generated under `output/l2-66136969/`.

- [ ] **Step 3: Verify output counts**

Run:

```bash
find output/l2-66136969/clips -maxdepth 1 -name '*.mp4' | wc -l
find output/l2-66136969/subtitles -maxdepth 1 -name '[0-9]*.srt' | wc -l
```

Expected: output counts match planned clip count.

- [ ] **Step 4: Spot-check the result**

Inspect a few clip durations and titles from the plan versus generated files.

- [ ] **Step 5: Report outcome**

Summarize what improved and any remaining weaknesses in the broadcaster-focused strategy.

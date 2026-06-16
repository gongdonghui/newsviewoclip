# Pause-Aware Boundary Snapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight pause-detection stage and use its results to snap clip boundaries toward natural speech pauses without changing the existing output schema or `mlx_whisper` transcription flow.

**Architecture:** Keep semantic clip selection as the primary strategy. Insert a new `ffmpeg silencedetect`-based pause analysis step after transcription, persist `pauses.json`, and apply pause-aware boundary snapping during clip-plan normalization for both DeepSeek and fallback clip paths. The implementation must degrade gracefully when pause detection fails and must preserve current cutting, subtitle export, and output directory behavior.

**Tech Stack:** Python 3.12, `ffmpeg` / `silencedetect`, `pytest`, existing `mlx_whisper`, `ffprobe`, `ffmpeg`, `argparse`, `pathlib`, `subprocess`, `logging`

---

## File Structure

- Create: `scripts/detect_pauses.py`
  - Single responsibility: invoke `ffmpeg silencedetect`, parse pauses, write `pauses.json`, and support reuse / `--force`.
- Modify: `scripts/run_pipeline.py`
  - Add pause detection between transcription and clip planning, persist `pauses.json`, and pass pause data into plan creation while degrading gracefully on failure.
- Modify: `scripts/build_clip_plan.py`
  - Add pause-aware snapping helpers and apply them to both DeepSeek-returned clips and fallback-generated clips before final normalization.
- Create: `tests/test_detect_pauses.py`
  - Unit coverage for parsing `silencedetect` output and payload creation.
- Modify: `tests/test_build_clip_plan.py`
  - Unit coverage for snapping to nearby pauses and preserving existing boundaries when no safe pause exists.
- Modify: `tests/test_run_pipeline.py`
  - Pipeline coverage for producing `pauses.json`, passing pause data into planning, and tolerating pause-detection failures.

## Task 1: Add Pause Detection Unit Tests

**Files:**
- Create: `tests/test_detect_pauses.py`

- [ ] **Step 1: Write the failing test for parsing `silencedetect` output**

```python
from scripts.detect_pauses import parse_silencedetect_output


def test_parse_silencedetect_output_extracts_pause_windows():
    stderr = """
    [silencedetect @ 0x0] silence_start: 12.48
    [silencedetect @ 0x0] silence_end: 13.21 | silence_duration: 0.73
    [silencedetect @ 0x0] silence_start: 28.04
    [silencedetect @ 0x0] silence_end: 28.82 | silence_duration: 0.78
    """

    assert parse_silencedetect_output(stderr, min_silence_seconds=0.45) == [
        {"start": 12.48, "end": 13.21, "duration": 0.73},
        {"start": 28.04, "end": 28.82, "duration": 0.78},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detect_pauses.py::test_parse_silencedetect_output_extracts_pause_windows -q`

Expected: FAIL because `scripts.detect_pauses` or `parse_silencedetect_output` does not exist yet.

- [ ] **Step 3: Add a second failing test for pause filtering**

```python
from scripts.detect_pauses import parse_silencedetect_output


def test_parse_silencedetect_output_filters_short_pauses():
    stderr = """
    [silencedetect @ 0x0] silence_start: 2.00
    [silencedetect @ 0x0] silence_end: 2.30 | silence_duration: 0.30
    [silencedetect @ 0x0] silence_start: 4.00
    [silencedetect @ 0x0] silence_end: 4.60 | silence_duration: 0.60
    """

    assert parse_silencedetect_output(stderr, min_silence_seconds=0.45) == [
        {"start": 4.0, "end": 4.6, "duration": 0.6},
    ]
```

- [ ] **Step 4: Run both tests to verify they fail correctly**

Run: `pytest tests/test_detect_pauses.py -q`

Expected: FAIL with import or missing-function errors, not syntax errors.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_detect_pauses.py
git commit -m "test: add pause detection parser coverage"
```

## Task 2: Implement `scripts/detect_pauses.py`

**Files:**
- Create: `scripts/detect_pauses.py`
- Test: `tests/test_detect_pauses.py`

- [ ] **Step 1: Write minimal parser implementation**

Implement:
- `parse_silencedetect_output(stderr: str, min_silence_seconds: float) -> list[dict[str, float]]`
- `build_pause_payload(source_audio: Path, noise_threshold_db: int, min_silence_seconds: float, pauses: list[dict[str, float]]) -> dict[str, object]`

- [ ] **Step 2: Run parser tests to verify they pass**

Run: `pytest tests/test_detect_pauses.py -q`

Expected: PASS for the parser-focused tests.

- [ ] **Step 3: Add a failing test for command execution and payload writing**

```python
from pathlib import Path

from scripts.detect_pauses import detect_pauses


def test_detect_pauses_writes_payload(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "source_audio.wav"
    audio_path.write_bytes(b"wav")
    output_path = tmp_path / "pauses.json"

    def fake_run(*args, **kwargs):
        class Result:
            stderr = "[silencedetect @ 0x0] silence_start: 1.0\n[silencedetect @ 0x0] silence_end: 1.7 | silence_duration: 0.7\n"
        return Result()

    monkeypatch.setattr("scripts.detect_pauses.subprocess.run", fake_run)

    payload = detect_pauses(audio_path, output_path, force=False)

    assert payload["pauses"] == [{"start": 1.0, "end": 1.7, "duration": 0.7}]
    assert output_path.exists()
```

- [ ] **Step 4: Run the new test and verify it fails**

Run: `pytest tests/test_detect_pauses.py::test_detect_pauses_writes_payload -q`

Expected: FAIL because `detect_pauses` behavior is not implemented yet.

- [ ] **Step 5: Implement the minimal `detect_pauses(...)` function**

Implementation requirements:
- invoke `ffmpeg` with `silencedetect`
- parse stderr
- write a JSON payload to the requested path
- reuse existing output unless `force=True`

- [ ] **Step 6: Run the full pause test file**

Run: `pytest tests/test_detect_pauses.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/detect_pauses.py tests/test_detect_pauses.py
git commit -m "feat: add ffmpeg-based pause detection"
```

## Task 3: Add Pause-Aware Boundary Snapping Tests

**Files:**
- Modify: `tests/test_build_clip_plan.py`
- Modify: `scripts/build_clip_plan.py`

- [ ] **Step 1: Write a failing test for snapping a clip end to a nearby pause**

```python
from scripts.build_clip_plan import normalize_clips


def test_normalize_clips_snaps_end_to_nearby_pause():
    clips = normalize_clips(
        [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 10.0,
                "end": 40.0,
                "summary": "Lead.",
                "keywords": [],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.5,
        padding_after=1.0,
        pauses=[{"start": 39.4, "end": 39.95, "duration": 0.55}],
    )

    assert clips[0]["end"] == "00:00:40.950"
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/test_build_clip_plan.py::test_normalize_clips_snaps_end_to_nearby_pause -q`

Expected: FAIL because `normalize_clips` does not accept or use `pauses` yet.

- [ ] **Step 3: Add a failing test for leaving boundaries unchanged when no pause qualifies**

```python
from scripts.build_clip_plan import normalize_clips


def test_normalize_clips_leaves_end_when_no_pause_is_nearby():
    clips = normalize_clips(
        [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 10.0,
                "end": 40.0,
                "summary": "Lead.",
                "keywords": [],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.5,
        padding_after=1.0,
        pauses=[{"start": 45.0, "end": 45.6, "duration": 0.6}],
    )

    assert clips[0]["end"] == "00:00:41.000"
```

- [ ] **Step 4: Run both tests and confirm failure is for missing pause-aware behavior**

Run: `pytest tests/test_build_clip_plan.py -q`

Expected: FAIL on the new pause-aware cases.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_build_clip_plan.py
git commit -m "test: cover pause-aware clip boundary snapping"
```

## Task 4: Implement Boundary Snapping

**Files:**
- Modify: `scripts/build_clip_plan.py`
- Test: `tests/test_build_clip_plan.py`

- [ ] **Step 1: Add pause-snapping helper functions**

Implement small focused helpers, for example:
- `snap_boundary_to_pause(...)`
- `snap_clip_end_to_pause(...)`
- `snap_clip_start_to_pause(...)`

Constraints:
- end snap window: back `1.2s`, forward `0.8s`
- start snap window: back `0.6s`, forward `0.8s`
- prefer backward snaps for end boundaries
- skip snapping when it would violate minimum duration or create excessive overlap

- [ ] **Step 2: Update `normalize_clips(...)` to accept optional `pauses`**

Keep `pauses` optional so existing callers stay valid while the pipeline is being integrated.

- [ ] **Step 3: Apply snapping before timestamp normalization**

Use pause data to refine raw clip boundaries before padding and string timestamp conversion.

- [ ] **Step 4: Run the build-clip-plan test file**

Run: `pytest tests/test_build_clip_plan.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_clip_plan.py tests/test_build_clip_plan.py
git commit -m "feat: add pause-aware clip boundary snapping"
```

## Task 5: Add Pipeline Integration Tests

**Files:**
- Modify: `tests/test_run_pipeline.py`
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Write a failing pipeline test for `pauses.json` creation**

Add a test that stubs pause detection and verifies:
- `output/metadata/pauses.json` is written
- pause payload is passed into clip planning

- [ ] **Step 2: Run the targeted test and verify it fails**

Run: `pytest tests/test_run_pipeline.py::test_run_pipeline_writes_pauses_and_uses_them_for_plan -q`

Expected: FAIL because the pipeline does not yet invoke pause detection.

- [ ] **Step 3: Add a failing pipeline test for graceful degradation**

Add a test where pause detection raises an exception and verify:
- pipeline still produces `clip_plan.json`
- execution continues instead of aborting

- [ ] **Step 4: Run both targeted tests and verify failure is due to missing integration**

Run: `pytest tests/test_run_pipeline.py -q`

Expected: FAIL on the new pause-related cases only.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_run_pipeline.py
git commit -m "test: add pause detection pipeline coverage"
```

## Task 6: Integrate Pause Detection Into the Pipeline

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `scripts/build_clip_plan.py`
- Modify: `tests/test_run_pipeline.py`
- Modify: `tests/test_build_clip_plan.py`
- Create/Use: `scripts/detect_pauses.py`

- [ ] **Step 1: Import and call pause detection after transcription**

Add a new output path:
- `directories["metadata"] / "pauses.json"`

Invoke pause detection against:
- `run_temp / "source_audio.wav"`

- [ ] **Step 2: Pass pause payload into both clip planning paths**

Requirements:
- DeepSeek path: `normalize_clips(..., pauses=...)`
- fallback path: `build_plan_preview(..., pauses=...)`

- [ ] **Step 3: Handle pause detection failures as warnings**

If pause analysis fails:
- log a warning
- continue with `pauses=[]`

- [ ] **Step 4: Run pipeline tests**

Run: `pytest tests/test_run_pipeline.py -q`

Expected: PASS.

- [ ] **Step 5: Run combined targeted regression**

Run: `pytest tests/test_detect_pauses.py tests/test_build_clip_plan.py tests/test_run_pipeline.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/detect_pauses.py scripts/run_pipeline.py scripts/build_clip_plan.py tests/test_detect_pauses.py tests/test_build_clip_plan.py tests/test_run_pipeline.py
git commit -m "feat: integrate pause-aware clip boundary snapping"
```

## Task 7: Manual Validation on Real News Inputs

**Files:**
- Modify: none required unless bugs are found
- Inspect: `output/<run-id>/metadata/clip_plan.md`, `output/<run-id>/metadata/pauses.json`

- [ ] **Step 1: Run a plan-only validation against a known problematic news file**

Run:

```bash
python3 scripts/run_pipeline.py --input ~/Downloads/l5.mp4 --mode news --language en --plan-only --force
```

Expected:
- `output/<run-id>/metadata/pauses.json` exists
- `clip_plan.md` is regenerated

- [ ] **Step 2: Inspect the first few clip endings against `pauses.json`**

Check that:
- clip endings align near detected pauses
- obvious anchor mid-thought cuts are reduced

- [ ] **Step 3: If the plan is materially better, run execute mode**

Run:

```bash
python3 scripts/run_pipeline.py --input ~/Downloads/l5.mp4 --mode news --language en --execute --force
```

Expected:
- clip files generated successfully
- subtitles still export correctly

- [ ] **Step 4: Commit only if code changed during validation**

```bash
git add <changed-files>
git commit -m "fix: tune pause-aware clipping after validation"
```

## Final Verification Checklist

- [ ] Run: `pytest tests/test_detect_pauses.py tests/test_build_clip_plan.py tests/test_run_pipeline.py -q`
- [ ] Confirm `pauses.json` is written during pipeline runs
- [ ] Confirm pause detection failure does not block clip generation
- [ ] Confirm clip output paths and subtitle output paths remain unchanged
- [ ] Confirm no schema changes to `clip_plan.json`, `clip_plan.csv`, or `clip_plan.md`

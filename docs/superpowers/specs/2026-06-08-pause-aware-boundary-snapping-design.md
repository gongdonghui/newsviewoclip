# Pause-Aware Boundary Snapping Design

## Summary

Improve clip boundary quality by adding a lightweight pause-detection stage to the current transcript-driven clipping pipeline.

The new behavior should:

- keep semantic clip selection as the primary decision maker
- use detected speech pauses as safe boundary hints
- prioritize fixing unnatural clip endings where the anchor is cut off mid-thought
- preserve the existing `mlx_whisper` transcription backend and output schema

## Goals

- Reduce cases where a clip ends while the anchor is still speaking.
- Snap clip boundaries to natural pauses when a suitable pause exists near the semantic boundary.
- Keep the implementation lightweight and compatible with the current pipeline.
- Allow both DeepSeek-generated clips and local fallback clips to benefit from the same pause data.

## Non-Goals

- No speaker diarization in this phase.
- No scene-detection integration in this phase.
- No replacement of `mlx_whisper`.
- No change to subtitle export format, cut execution, or output directory layout.
- No attempt to make pause detection the primary segmentation strategy.

## Chosen Approach

Add a dedicated pause-analysis step after audio extraction and before clip planning.

Semantic planning remains responsible for deciding what content belongs in a clip. Pause data is only used to improve where the chosen clip starts or ends.

This is preferred over introducing scene detection or diarization first because the immediate problem is boundary naturalness, not speaker classification or visual shot alignment.

## Pipeline Placement

Current pipeline:

- probe video
- extract audio
- transcribe audio
- generate clip plan
- cut clips
- export clip subtitles

Proposed pipeline:

- probe video
- extract audio
- transcribe audio
- detect pauses
- generate clip plan with pause-aware boundary snapping
- cut clips
- export clip subtitles

## Pause Detection Strategy

### Detection mechanism

Use `ffmpeg` with the `silencedetect` filter against the extracted mono WAV file.

This keeps the implementation simple because:

- `ffmpeg` is already a required dependency
- no new heavy speech model is introduced
- the output is easy to log, inspect, and reproduce

### Initial thresholds

- noise threshold: `-35dB`
- minimum silence duration: `0.45s`

These values are intentionally conservative for broadcast-news audio, where meaningful pauses are often short but still distinct.

### Output artifact

Write pause analysis to:

- `output/metadata/pauses.json`

Suggested payload:

```json
{
  "source_audio": "temp/run-id/source_audio.wav",
  "noise_threshold_db": -35,
  "min_silence_seconds": 0.45,
  "pauses": [
    {"start": 12.48, "end": 13.21, "duration": 0.73},
    {"start": 28.04, "end": 28.82, "duration": 0.78}
  ]
}
```

## Boundary Snapping Rules

### Core principle

Semantic selection determines the candidate clip.

Pause data only adjusts the final boundary to make the clip sound more natural.

### End-boundary snapping

This is the highest-priority improvement.

For each selected clip end:

- look backward up to `1.2s`
- look forward up to `0.8s`
- prefer snapping to a nearby pause rather than keeping the raw semantic end
- prefer a backward snap over a forward snap when both are valid

This bias avoids dragging the next anchor sentence into the clip.

### Start-boundary snapping

This should be lighter and more conservative than end snapping.

For each selected clip start:

- look backward up to `0.6s`
- look forward up to `0.8s`
- snap only when the result remains coherent and does not pull in too much of the previous thought

### Safety conditions

Do not apply a snap if it:

- makes the clip shorter than the configured minimum duration
- causes excessive overlap with a neighboring clip
- moves the boundary to a pause clearly outside the intended semantic unit

If no suitable pause is found, keep the original semantic boundary.

## Code Changes

### 1. Add `scripts/detect_pauses.py`

Responsibilities:

- run `ffmpeg silencedetect`
- parse the silence log output
- filter pauses shorter than the configured threshold
- write `pauses.json`
- support file reuse unless `--force` is requested

### 2. Update `scripts/run_pipeline.py`

Add a new pause-analysis step between transcription and clip planning.

Responsibilities:

- invoke pause detection on the extracted WAV
- persist `output/metadata/pauses.json`
- pass pause data into clip planning
- degrade gracefully if pause detection fails

Failure in pause detection should log a warning and continue with the original semantic behavior.

### 3. Update `scripts/build_clip_plan.py`

Add pause-aware boundary utilities, for example:

- `snap_clip_end_to_pause(...)`
- `snap_clip_start_to_pause(...)`

These utilities should be applied to:

- fallback-generated raw clips
- DeepSeek-generated raw clips before final normalization

### 4. Keep `scripts/analyze_clips.py` unchanged for phase 1

Pause awareness in this phase should be a boundary-postprocessing concern, not a prompt-design concern.

## Testing

### New tests

Add:

- `tests/test_detect_pauses.py`

Cover:

- parsing `silencedetect` output
- filtering by minimum duration
- writing pause payload structure

### Update existing tests

Extend:

- `tests/test_build_clip_plan.py`
- `tests/test_run_pipeline.py`

Cover:

- clip end snapping to the nearest valid pause
- no snapping when no valid pause exists
- protection against snapping below minimum duration
- pipeline emits `pauses.json`
- pipeline still works when pause detection is unavailable or fails

## Risks

- Broadcast music, stingers, and noisy field reports may create false pause candidates.
- Very short broadcast breathing gaps may still not be safe cut points.
- Over-aggressive snapping could shift a clip away from the intended payoff.

These risks are why pause detection should remain a secondary boundary-refinement signal, not the primary segmentation strategy.

## Success Criteria

- Fewer clips end in the middle of the anchor's spoken thought.
- More clip endings align with natural pauses in speech.
- DeepSeek and fallback paths both benefit from the same pause-aware snapping logic.
- The existing output schema and `mlx_whisper` pipeline remain intact.
- If pause detection fails, the pipeline still produces a valid clip plan and clip outputs.

# Short Broadcaster-Focused Clip Strategy Design

## Summary

Change the clip planning strategy from broad, semantically complete news-story segments to short, concentrated broadcaster-led highlights.

The new default behavior is:

- target clip duration: 15-45 seconds
- preference: anchor first
- allowed continuation: brief reporter or soundbite continuation only when it sharpens the point
- rejection bias: avoid long correspondent packages, multi-topic bundles, and background-heavy context blocks

## Goals

- Produce shorter clips that are more suitable for social or highlight consumption.
- Focus clip selection on the broadcaster's delivery and summary lines.
- Increase the number of concentrated clips rather than generating a few long story bundles.
- Preserve the existing transcription backend: `mlx_whisper`.

## Non-Goals

- No speaker diarization or formal speaker-classification subsystem.
- No UI changes.
- No changes to clip cutting, subtitle export, or concurrency/isolation behavior.
- No new model provider or transcription backend changes.

## Chosen Approach

Use prompt tightening plus fallback heuristic tightening.

This keeps the implementation surgical while improving both:

- LLM-driven clip analysis through stronger segmentation instructions
- local fallback behavior when DeepSeek analysis is unavailable or rejected

This is preferred over a prompt-only change because the fallback path would otherwise remain biased toward long, story-complete clips.

## Desired Behavior

### Clip shape

- Default clip duration should be 15-45 seconds.
- The planner should prefer more clips that are tighter and more focused.
- The planner should treat a concise anchor setup plus payoff as a strong clip candidate.

### Content bias

- Prefer broadcaster-led openings.
- Prefer clips where the anchor states the key point directly.
- Allow a short reporter or soundbite continuation only when it improves clarity or delivers the payoff line.

### Content to avoid

- Long correspondent packages.
- Wide background explainers when a narrower excerpt works.
- Combined multi-topic blocks.
- Weak intros with no clear payoff.

## Implementation

### 1. Tighten DeepSeek analysis prompt

Update `scripts/analyze_clips.py` so the system prompt explicitly requires:

- 15-45 second clips
- broadcaster-first selection
- short continuation only when it sharpens the point
- preference for tighter, more numerous clips over long story packages
- rejection of diffuse background-heavy and long package-style segments

The output schema remains unchanged.

### 2. Tighten fallback defaults

Update `scripts/run_pipeline.py` constants from:

- min: 20
- target: 60
- max: 180

to:

- min: 15
- target: 30
- max: 45

Also update the printed strategy summary so it matches the new behavior.

### 3. Tighten fallback grouping

Update `scripts/build_clip_plan.py` fallback grouping so it:

- flushes groups earlier
- avoids merging into long narrative bundles
- prefers short payoff windows
- still preserves the existing padding behavior

The fallback path should remain simple and heuristic-driven, not speaker-aware.

## Verification

### Unit tests

- Add a test that verifies the DeepSeek prompt includes:
  - 15-45 second duration guidance
  - broadcaster-first preference
  - brief continuation rule
  - tighter/more numerous clips preference
- Update fallback clip-plan tests to reflect shorter default clip windows.

### Regression checks

- Existing run-pipeline tests must still pass.
- Existing concurrency and isolated-output behavior must remain unchanged.

### Practical validation

Re-run `~/Downloads/l2.mp4` in `--plan-only --force` first and inspect the generated `clip_plan.md`.

Expected change:

- more clips
- shorter average duration
- more broadcaster-led intros
- fewer long, complete-story packages

Only after the plan quality looks materially better should the execute path be re-run.

## Success Criteria

- Median clip duration drops substantially versus the previous long-story strategy.
- Clip titles and summaries represent tighter single points.
- Broadcaster-led clip openings appear more frequently.
- The transcription backend remains `mlx_whisper`.

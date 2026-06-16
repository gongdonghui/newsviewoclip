# L1 Semantic Clip Plan Design

## Summary

Build a local-first, plan-only video analysis pipeline for `/Users/gongshuai/Downloads/l1.mp4`.
The pipeline assumes an English news video, transcribes audio with the local MLX Whisper model `mlx-community/whisper-large-v3-mlx`, uses DeepSeek to perform semantic clip analysis over the transcript, and writes a human-reviewable clip plan without exporting any clipped videos.

## Goals

- Produce source metadata for the input video.
- Produce reusable transcription artifacts with timestamps.
- Produce a semantic clip plan in JSON, CSV, and Markdown.
- Keep the source video untouched.
- Prefer local processing, with DeepSeek used only for transcript-level semantic analysis.

## Non-Goals

- Cutting or exporting video clips.
- Burning subtitles into video.
- Generating thumbnails.
- Scene detection or frame-based analysis in v1.
- Supporting arbitrary content types beyond the first-pass English news workflow.

## Inputs And Defaults

- Input video: `/Users/gongshuai/Downloads/l1.mp4`
- Working directory: `/Users/gongshuai/workspace/english-video-clip`
- Mode: `news`
- Language: `en`
- Execution mode: `plan-only`
- Minimum clip duration: `20` seconds
- Target clip duration: `60` to `120` seconds
- Maximum clip duration: `180` seconds
- Padding before clip start: `0.5` seconds
- Padding after clip end: `1.0` second

## Outputs

The pipeline writes the following artifacts under the project directory:

- `output/metadata/source_probe.json`
- `output/metadata/transcript.json`
- `output/metadata/transcript.txt`
- `output/metadata/clip_plan.json`
- `output/metadata/clip_plan.csv`
- `output/metadata/clip_plan.md`
- `output/subtitles/source.srt`
- `output/subtitles/source.vtt`
- `temp/source_audio.wav`
- `output/logs/pipeline.log`

## Architecture

The implementation is a small set of focused Python scripts coordinated by `scripts/run_pipeline.py`.
Each step writes durable artifacts so reruns can skip completed work unless `--force` is provided.
The primary flow is:

1. Probe the video with `ffprobe`.
2. Extract mono 16kHz WAV audio with `ffmpeg`.
3. Transcribe audio locally with `mlx_whisper` using `mlx-community/whisper-large-v3-mlx`.
4. Ask DeepSeek to identify semantically complete news clips from the transcript.
5. Normalize and validate the returned clip suggestions locally.
6. Write the clip plan in JSON, CSV, and Markdown.

If the DeepSeek call is unavailable or malformed, the pipeline falls back to local heuristic grouping based on transcript timing and sentence boundaries.

## Components

### `scripts/common.py`

Shared helpers for:

- Project paths and output directory creation
- `.env` loading
- Logging setup
- Timestamp conversion
- JSON, CSV, and Markdown serialization
- External command execution with `subprocess.run(..., check=True)`

### `scripts/probe_video.py`

Responsibilities:

- Run `ffprobe` against the input video
- Save the raw normalized metadata to `output/metadata/source_probe.json`
- Return basic fields used downstream:
  - duration
  - resolution
  - frame rate
  - codecs
  - sample rate
  - file size

### `scripts/extract_audio.py`

Responsibilities:

- Extract audio from the source video to `temp/source_audio.wav`
- Use mono, 16kHz, PCM S16LE
- Reuse the extracted file if present unless `--force` is set

### `scripts/transcribe.py`

Responsibilities:

- Load the local model `mlx-community/whisper-large-v3-mlx`
- Force transcription language to English
- Produce segment-level transcript data with start, end, and text
- Write:
  - `output/metadata/transcript.json`
  - `output/metadata/transcript.txt`
  - `output/subtitles/source.srt`
  - `output/subtitles/source.vtt`

Transcript JSON schema:

```json
{
  "source_video": "/Users/gongshuai/Downloads/l1.mp4",
  "language": "en",
  "model": "mlx-community/whisper-large-v3-mlx",
  "segments": [
    {
      "id": 1,
      "start": 12.45,
      "end": 18.92,
      "text": "Transcript text."
    }
  ]
}
```

### `scripts/analyze_clips.py`

Responsibilities:

- Load `DEEPSEEK_API_KEY` from `.env`
- Send source metadata and transcript segments to DeepSeek
- Request semantic clip candidates for English news segmentation
- Ask for structured JSON with:
  - `clip_id`
  - `title`
  - `start`
  - `end`
  - `summary`
  - `keywords`
  - `reason`
  - `confidence`

DeepSeek is used on transcript text and metadata, not on raw video bytes.

### `scripts/build_clip_plan.py`

Responsibilities:

- Validate clip candidates from DeepSeek
- Clamp boundaries to video duration
- Apply start/end padding while preserving valid ranges
- Reject empty or inverted ranges
- Merge or discard clips that violate minimum duration
- Fall back to local heuristic grouping when DeepSeek is unavailable or invalid
- Write:
  - `output/metadata/clip_plan.json`
  - `output/metadata/clip_plan.csv`
  - `output/metadata/clip_plan.md`

Final clip plan item shape:

```json
{
  "clip_id": "clip_001",
  "title": "Market update and inflation outlook",
  "start": "00:01:12.300",
  "end": "00:02:05.800",
  "duration_seconds": 53.5,
  "summary": "The anchor covers the latest market reaction and links it to the inflation report.",
  "keywords": ["market", "inflation", "stocks"],
  "reason": "Topic is self-contained and ends on a complete handoff.",
  "confidence": 0.92,
  "output_file": "output/clips/clip_001.mp4"
}
```

In `plan-only` mode, `output_file` remains planned metadata only and no video file is created.

### `scripts/run_pipeline.py`

Responsibilities:

- Parse CLI arguments with `argparse`
- Create directory structure
- Run the pipeline steps in order
- Support `--plan-only` and `--force`
- Print the project-required execution summary

Expected primary command:

```bash
python3 scripts/run_pipeline.py \
  --input /Users/gongshuai/Downloads/l1.mp4 \
  --mode news \
  --language en \
  --plan-only
```

## DeepSeek Integration

The pipeline uses the `openai` Python client against DeepSeek-compatible chat/completions endpoints, configured through `.env`.
The minimum required variable is:

- `DEEPSEEK_API_KEY`

Optional variables may be supported if present:

- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`

If the API key is missing, the pipeline logs the condition and falls back to local heuristic planning instead of failing the entire run.

## Heuristic Fallback

Fallback behavior groups transcript segments using a conservative local strategy:

- Prefer sentence-ending punctuation
- Prefer pauses larger than a configurable threshold
- Split on strong topic-shift markers when obvious
- Keep clips between the configured minimum and maximum durations
- Bias toward preserving more context rather than aggressive splitting

This fallback exists so the pipeline remains usable without a live DeepSeek response.

## Error Handling

- Missing input video stops the run with a clear error.
- Missing `ffmpeg`, `ffprobe`, or required Python modules stops the run with a clear error.
- Existing artifacts are reused unless `--force` is passed.
- DeepSeek failures degrade to local heuristic planning when possible.
- Malformed DeepSeek JSON is treated as an analysis failure and triggers fallback.
- All major steps log the command or operation being executed.

## Testing Strategy

Unit tests will cover:

- Timestamp formatting and parsing
- Transcript serialization
- Clip range normalization
- Markdown/CSV plan rendering
- Fallback grouping behavior from a transcript fixture

One integration-style test will cover:

- Running clip-plan generation from a small transcript fixture without requiring the real video or network access

Manual verification will cover:

- Running the full pipeline in `plan-only` mode against `/Users/gongshuai/Downloads/l1.mp4`
- Confirming all required artifacts are produced

## Success Criteria

The implementation is complete when:

- The pipeline runs in `plan-only` mode for `/Users/gongshuai/Downloads/l1.mp4`.
- It writes source metadata, transcript artifacts, and clip plan artifacts.
- It does not export any final clips.
- It leaves the source video unchanged.
- It can still produce a clip plan when DeepSeek analysis is unavailable.

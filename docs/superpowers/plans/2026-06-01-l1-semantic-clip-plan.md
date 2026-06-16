# L1 Semantic Clip Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first, plan-only pipeline that probes `/Users/gongshuai/Downloads/l1.mp4`, transcribes it with local MLX Whisper, analyzes transcript segments with DeepSeek, and writes clip-plan artifacts without exporting clips.

**Architecture:** The pipeline is a set of small Python scripts under `scripts/` with shared helpers in `scripts/common.py`. `scripts/run_pipeline.py` orchestrates probe, audio extraction, transcription, DeepSeek analysis, fallback planning, and artifact writing; `pytest` covers local helpers and clip-plan normalization without requiring the real video or network.

**Tech Stack:** Python 3.12, `argparse`, `pathlib`, `subprocess`, `logging`, `pytest`, `mlx_whisper`, `openai`, `ffmpeg`, `ffprobe`

---

### Task 1: Create Shared Helpers And Test Harness

**Files:**
- Create: `scripts/common.py`
- Create: `tests/test_common.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from scripts.common import (
    format_seconds,
    load_env_file,
    seconds_to_timestamp,
    timestamp_to_seconds,
)


def test_timestamp_round_trip_with_milliseconds():
    value = 72.345
    timestamp = seconds_to_timestamp(value)
    assert timestamp == "00:01:12.345"
    assert timestamp_to_seconds(timestamp) == 72.345


def test_format_seconds_uses_single_decimal_for_summary():
    assert format_seconds(53.54) == "53.5"


def test_load_env_file_reads_simple_key_values(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=test-key\nDEEPSEEK_MODEL=deepseek-chat\n", encoding="utf-8")

    values = load_env_file(env_file)

    assert values["DEEPSEEK_API_KEY"] == "test-key"
    assert values["DEEPSEEK_MODEL"] == "deepseek-chat"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_common.py -q`
Expected: `ModuleNotFoundError` or import failure for `scripts.common`

- [ ] **Step 3: Write the minimal implementation**

```python
from __future__ import annotations

from pathlib import Path


def seconds_to_timestamp(value: float) -> str:
    total_milliseconds = int(round(value * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def timestamp_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    whole_seconds, milliseconds = seconds.split(".")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(whole_seconds)
        + int(milliseconds) / 1000
    )


def format_seconds(value: float) -> str:
    return f"{value:.1f}"


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        values[key.strip()] = raw_value.strip()
    return values
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_common.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/common.py tests/test_common.py
git commit -m "test: add shared helper coverage"
```

### Task 2: Add Transcript And Clip-Plan Serialization Helpers

**Files:**
- Modify: `scripts/common.py`
- Create: `tests/test_serialization.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

from scripts.common import write_json, write_markdown_clip_plan


def test_write_json_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "data.json"

    write_json(path, {"language": "en"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"language": "en"}


def test_write_markdown_clip_plan_renders_table(tmp_path: Path):
    path = tmp_path / "clip_plan.md"
    clips = [
        {
            "clip_id": "clip_001",
            "title": "Opening headline",
            "start": "00:00:00.000",
            "end": "00:01:00.000",
            "duration_seconds": 60.0,
            "summary": "Anchor introduces the lead story.",
            "reason": "Self-contained introduction.",
        }
    ]

    write_markdown_clip_plan(path, clips)

    content = path.read_text(encoding="utf-8")
    assert "| 编号 | 标题 | 开始时间 | 结束时间 | 时长 | 摘要 | 建议理由 |" in content
    assert "clip_001" in content
    assert "Opening headline" in content
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_serialization.py -q`
Expected: import failure for missing writer helpers

- [ ] **Step 3: Write the minimal implementation**

```python
import json
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_clip_plan(path: Path, clips: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| 编号 | 标题 | 开始时间 | 结束时间 | 时长 | 摘要 | 建议理由 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for clip in clips:
        lines.append(
            "| {clip_id} | {title} | {start} | {end} | {duration_seconds:.1f} 秒 | {summary} | {reason} |".format(
                **clip
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_serialization.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/common.py tests/test_serialization.py
git commit -m "test: cover plan serialization helpers"
```

### Task 3: Implement Probe And Audio Extraction Utilities

**Files:**
- Create: `scripts/probe_video.py`
- Create: `scripts/extract_audio.py`
- Create: `tests/test_probe_audio.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from scripts.extract_audio import build_extract_audio_command
from scripts.probe_video import normalize_probe_payload


def test_normalize_probe_payload_extracts_expected_fields():
    payload = {
        "format": {"duration": "125.5", "size": "2048"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30000/1001", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }

    normalized = normalize_probe_payload(Path("/tmp/video.mp4"), payload)

    assert normalized["duration_seconds"] == 125.5
    assert normalized["resolution"] == "1920x1080"
    assert normalized["video_codec"] == "h264"
    assert normalized["audio_codec"] == "aac"
    assert normalized["audio_sample_rate"] == 48000


def test_build_extract_audio_command_targets_wav_output():
    command = build_extract_audio_command(Path("input.mp4"), Path("temp/source_audio.wav"))
    assert command == [
        "ffmpeg",
        "-i",
        "input.mp4",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "temp/source_audio.wav",
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_probe_audio.py -q`
Expected: import failure for missing probe/extraction modules

- [ ] **Step 3: Write the minimal implementation**

```python
from pathlib import Path


def normalize_probe_payload(source: Path, payload: dict[str, object]) -> dict[str, object]:
    streams = payload["streams"]
    video_stream = next(stream for stream in streams if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in streams if stream["codec_type"] == "audio")
    return {
        "source_video": str(source),
        "duration_seconds": float(payload["format"]["duration"]),
        "file_size_bytes": int(payload["format"]["size"]),
        "resolution": f"{video_stream['width']}x{video_stream['height']}",
        "frame_rate": video_stream["r_frame_rate"],
        "video_codec": video_stream["codec_name"],
        "audio_codec": audio_stream["codec_name"],
        "audio_sample_rate": int(audio_stream["sample_rate"]),
    }


def build_extract_audio_command(source: Path, output: Path) -> list[str]:
    return [
        "ffmpeg",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_probe_audio.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_video.py scripts/extract_audio.py tests/test_probe_audio.py
git commit -m "test: cover probe and audio extraction helpers"
```

### Task 4: Implement Transcript Conversion And Subtitle Writers

**Files:**
- Create: `scripts/transcribe.py`
- Create: `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing tests**

```python
from scripts.transcribe import build_transcript_payload, render_srt_entries, render_vtt_entries


def test_build_transcript_payload_preserves_segments():
    payload = build_transcript_payload(
        "/tmp/video.mp4",
        "en",
        "mlx-community/whisper-large-v3-mlx",
        [(0.0, 1.25, "Hello world.")],
    )

    assert payload["language"] == "en"
    assert payload["model"] == "mlx-community/whisper-large-v3-mlx"
    assert payload["segments"][0]["text"] == "Hello world."


def test_render_srt_and_vtt_include_expected_timestamps():
    segments = [{"id": 1, "start": 0.0, "end": 1.25, "text": "Hello world."}]

    srt = render_srt_entries(segments)
    vtt = render_vtt_entries(segments)

    assert "00:00:00,000 --> 00:00:01,250" in srt
    assert "WEBVTT" in vtt
    assert "00:00:00.000 --> 00:00:01.250" in vtt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_transcribe.py -q`
Expected: import failure for missing transcription helpers

- [ ] **Step 3: Write the minimal implementation**

```python
from scripts.common import seconds_to_timestamp


def build_transcript_payload(
    source_video: str,
    language: str,
    model_name: str,
    raw_segments: list[tuple[float, float, str]],
) -> dict[str, object]:
    segments = []
    for index, (start, end, text) in enumerate(raw_segments, start=1):
        segments.append({"id": index, "start": start, "end": end, "text": text.strip()})
    return {
        "source_video": source_video,
        "language": language,
        "model": model_name,
        "segments": segments,
    }


def render_srt_entries(segments: list[dict[str, object]]) -> str:
    blocks = []
    for segment in segments:
        start = seconds_to_timestamp(segment["start"]).replace(".", ",")
        end = seconds_to_timestamp(segment["end"]).replace(".", ",")
        blocks.append(f"{segment['id']}\n{start} --> {end}\n{segment['text']}")
    return "\n\n".join(blocks) + "\n"


def render_vtt_entries(segments: list[dict[str, object]]) -> str:
    blocks = ["WEBVTT\n"]
    for segment in segments:
        start = seconds_to_timestamp(segment["start"])
        end = seconds_to_timestamp(segment["end"])
        blocks.append(f"{start} --> {end}\n{segment['text']}\n")
    return "\n".join(blocks)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_transcribe.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/transcribe.py tests/test_transcribe.py
git commit -m "test: cover transcript and subtitle rendering"
```

### Task 5: Implement DeepSeek Request Builder And Response Parser

**Files:**
- Create: `scripts/analyze_clips.py`
- Create: `tests/test_analyze_clips.py`

- [ ] **Step 1: Write the failing tests**

```python
from scripts.analyze_clips import build_analysis_messages, parse_analysis_response


def test_build_analysis_messages_mentions_news_segmentation():
    messages = build_analysis_messages(
        {"duration_seconds": 120.0, "resolution": "1920x1080"},
        [{"id": 1, "start": 0.0, "end": 10.0, "text": "Opening headline."}],
    )

    assert "English news video" in messages[0]["content"]
    assert "Opening headline." in messages[1]["content"]


def test_parse_analysis_response_extracts_json_payload():
    content = '{"clips":[{"clip_id":"clip_001","title":"Opening headline","start":0.0,"end":60.0,"summary":"Lead story.","keywords":["headline"],"reason":"Complete segment.","confidence":0.9}]}'

    clips = parse_analysis_response(content)

    assert clips[0]["clip_id"] == "clip_001"
    assert clips[0]["end"] == 60.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_analyze_clips.py -q`
Expected: import failure for missing DeepSeek helpers

- [ ] **Step 3: Write the minimal implementation**

```python
import json


def build_analysis_messages(
    probe_metadata: dict[str, object],
    transcript_segments: list[dict[str, object]],
) -> list[dict[str, str]]:
    system_message = (
        "You are segmenting an English news video into semantically complete clips. "
        "Return strict JSON with a top-level clips array."
    )
    user_lines = [
        "Video metadata:",
        json.dumps(probe_metadata, ensure_ascii=False),
        "Transcript segments:",
        json.dumps(transcript_segments, ensure_ascii=False),
    ]
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def parse_analysis_response(content: str) -> list[dict[str, object]]:
    payload = json.loads(content)
    return payload["clips"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_analyze_clips.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_clips.py tests/test_analyze_clips.py
git commit -m "test: cover DeepSeek analysis helpers"
```

### Task 6: Implement Clip Normalization And Heuristic Fallback

**Files:**
- Create: `scripts/build_clip_plan.py`
- Create: `tests/test_build_clip_plan.py`

- [ ] **Step 1: Write the failing tests**

```python
from scripts.build_clip_plan import build_fallback_clips, normalize_clips


def test_normalize_clips_applies_padding_and_duration():
    clips = normalize_clips(
        [{"clip_id": "clip_001", "title": "Lead", "start": 10.0, "end": 40.0, "summary": "Lead.", "keywords": ["lead"], "reason": "Complete.", "confidence": 0.9}],
        duration_seconds=120.0,
        padding_before=0.5,
        padding_after=1.0,
    )

    assert clips[0]["start"] == "00:00:09.500"
    assert clips[0]["end"] == "00:00:41.000"
    assert clips[0]["duration_seconds"] == 31.5


def test_build_fallback_clips_groups_segments_until_target_duration():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 20.0, "text": "Opening headline."},
            {"id": 2, "start": 20.5, "end": 45.0, "text": "Reporter adds details."},
            {"id": 3, "start": 60.0, "end": 90.0, "text": "Second story begins."},
        ]
    }

    clips = build_fallback_clips(
        transcript,
        min_clip_seconds=20,
        target_clip_seconds=60,
        max_clip_seconds=180,
    )

    assert len(clips) == 2
    assert clips[0]["start"] == 0.0
    assert clips[0]["end"] == 45.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_build_clip_plan.py -q`
Expected: import failure for missing plan builder

- [ ] **Step 3: Write the minimal implementation**

```python
from scripts.common import seconds_to_timestamp


def normalize_clips(
    raw_clips: list[dict[str, object]],
    duration_seconds: float,
    padding_before: float,
    padding_after: float,
) -> list[dict[str, object]]:
    normalized = []
    for index, clip in enumerate(raw_clips, start=1):
        start = max(0.0, float(clip["start"]) - padding_before)
        end = min(duration_seconds, float(clip["end"]) + padding_after)
        duration = round(end - start, 3)
        normalized.append(
            {
                "clip_id": clip.get("clip_id", f"clip_{index:03d}"),
                "title": clip["title"],
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
                "duration_seconds": duration,
                "summary": clip["summary"],
                "keywords": clip.get("keywords", []),
                "reason": clip["reason"],
                "confidence": float(clip.get("confidence", 0.0)),
                "output_file": f"output/clips/clip_{index:03d}.mp4",
            }
        )
    return normalized


def build_fallback_clips(
    transcript: dict[str, object],
    min_clip_seconds: int,
    target_clip_seconds: int,
    max_clip_seconds: int,
) -> list[dict[str, object]]:
    clips = []
    current = []
    for segment in transcript["segments"]:
        current.append(segment)
        start = current[0]["start"]
        end = current[-1]["end"]
        duration = end - start
        if duration >= target_clip_seconds or duration >= max_clip_seconds:
            clips.append(
                {
                    "title": current[0]["text"][:60],
                    "start": start,
                    "end": end,
                    "summary": " ".join(item["text"] for item in current),
                    "keywords": [],
                    "reason": "Fallback semantic grouping.",
                    "confidence": 0.0,
                }
            )
            current = []
    if current:
        clips.append(
            {
                "title": current[0]["text"][:60],
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "summary": " ".join(item["text"] for item in current),
                "keywords": [],
                "reason": "Fallback semantic grouping.",
                "confidence": 0.0,
            }
        )
    return clips
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_build_clip_plan.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_clip_plan.py tests/test_build_clip_plan.py
git commit -m "test: cover clip normalization and fallback"
```

### Task 7: Implement Pipeline Entry Point And End-To-End Fixture Test

**Files:**
- Create: `scripts/run_pipeline.py`
- Create: `tests/test_run_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from scripts.run_pipeline import build_argument_parser


def test_argument_parser_supports_plan_only():
    parser = build_argument_parser()
    args = parser.parse_args(["--input", "/tmp/video.mp4", "--mode", "news", "--language", "en", "--plan-only"])

    assert str(args.input) == "/tmp/video.mp4"
    assert args.mode == "news"
    assert args.language == "en"
    assert args.plan_only is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_run_pipeline.py -q`
Expected: import failure for missing pipeline entry point

- [ ] **Step 3: Write the minimal implementation**

```python
import argparse
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", default="news")
    parser.add_argument("--language", default="en")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_run_pipeline.py -q`
Expected: `1 passed`

- [ ] **Step 5: Expand to the fixture-driven integration test**

```python
import json
from pathlib import Path

from scripts.build_clip_plan import build_fallback_clips, normalize_clips


def test_fixture_transcript_produces_clip_plan():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 30.0, "text": "Opening headline."},
            {"id": 2, "start": 31.0, "end": 70.0, "text": "Reporter package."},
        ]
    }

    raw_clips = build_fallback_clips(transcript, 20, 60, 180)
    clips = normalize_clips(raw_clips, 120.0, 0.5, 1.0)

    assert clips[0]["clip_id"] == "clip_001"
    assert clips[0]["start"] == "00:00:00.000"
    assert clips[0]["end"] == "00:01:11.000"
```

- [ ] **Step 6: Run the full local test suite**

Run: `pytest tests -q`
Expected: `15 passed`

- [ ] **Step 7: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "test: add pipeline entrypoint coverage"
```

### Task 8: Wire Real Orchestration And Manual Verification

**Files:**
- Modify: `scripts/common.py`
- Modify: `scripts/probe_video.py`
- Modify: `scripts/extract_audio.py`
- Modify: `scripts/transcribe.py`
- Modify: `scripts/analyze_clips.py`
- Modify: `scripts/build_clip_plan.py`
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Implement real subprocess orchestration and artifact writing**

```python
def run_pipeline(args: argparse.Namespace) -> int:
    probe_metadata = probe_video(args.input, force=args.force)
    audio_path = extract_audio(args.input, force=args.force)
    transcript = transcribe_audio(audio_path, args.input, language=args.language, force=args.force)
    raw_clips = analyze_or_fallback(
        probe_metadata=probe_metadata,
        transcript=transcript,
        env_path=Path(".env"),
    )
    clips = normalize_clips(
        raw_clips,
        duration_seconds=probe_metadata["duration_seconds"],
        padding_before=0.5,
        padding_after=1.0,
    )
    write_clip_plan_outputs(clips)
    return 0
```

- [ ] **Step 2: Run the full test suite after wiring**

Run: `pytest tests -q`
Expected: `15 passed`

- [ ] **Step 3: Run the real plan-only pipeline against the target video**

Run:

```bash
python3 scripts/run_pipeline.py \
  --input /Users/gongshuai/Downloads/l1.mp4 \
  --mode news \
  --language en \
  --plan-only
```

Expected:
- exit code `0`
- generated files under `output/metadata`, `output/subtitles`, `output/logs`, and `temp/source_audio.wav`
- no files created under `output/clips`

- [ ] **Step 4: Verify expected artifacts exist**

Run:

```bash
ls output/metadata/source_probe.json \
   output/metadata/transcript.json \
   output/metadata/transcript.txt \
   output/metadata/clip_plan.json \
   output/metadata/clip_plan.csv \
   output/metadata/clip_plan.md \
   output/subtitles/source.srt \
   output/subtitles/source.vtt \
   temp/source_audio.wav
```

Expected: all paths listed with no `No such file or directory` errors

- [ ] **Step 5: Commit**

```bash
git add scripts tests
git commit -m "feat: implement semantic clip planning pipeline"
```

from __future__ import annotations

from pathlib import Path

from scripts.common import timestamp_to_seconds
from scripts.transcribe import render_srt_entries, render_vtt_entries


def build_clip_transcript_segments(
    transcript: dict[str, object],
    clip_start_seconds: float,
    clip_end_seconds: float,
) -> list[dict[str, object]]:
    clip_segments: list[dict[str, object]] = []
    for segment in transcript["segments"]:
        segment_start = max(float(segment["start"]), clip_start_seconds)
        segment_end = min(float(segment["end"]), clip_end_seconds)
        if segment_end <= segment_start:
            continue
        clip_segments.append(
            {
                "id": len(clip_segments) + 1,
                "start": round(segment_start - clip_start_seconds, 3),
                "end": round(segment_end - clip_start_seconds, 3),
                "text": str(segment["text"]),
            }
        )
    return clip_segments


def export_clip_subtitles(
    clip: dict[str, object],
    transcript: dict[str, object],
    subtitles_dir: Path,
) -> None:
    subtitles_dir.mkdir(parents=True, exist_ok=True)
    clip_start_seconds = timestamp_to_seconds(str(clip["start"]))
    clip_end_seconds = timestamp_to_seconds(str(clip["end"]))
    clip_segments = build_clip_transcript_segments(
        transcript,
        clip_start_seconds=clip_start_seconds,
        clip_end_seconds=clip_end_seconds,
    )
    clip_id = str(clip["clip_id"])
    (subtitles_dir / f"{clip_id}.srt").write_text(
        render_srt_entries(clip_segments),
        encoding="utf-8",
    )
    (subtitles_dir / f"{clip_id}.vtt").write_text(
        render_vtt_entries(clip_segments),
        encoding="utf-8",
    )

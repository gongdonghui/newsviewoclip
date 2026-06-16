from pathlib import Path

from scripts.export_clip_subtitles import (
    build_clip_transcript_segments,
    export_clip_subtitles,
)


def test_build_clip_transcript_segments_rebases_and_clamps_overlaps():
    transcript = {
        "segments": [
            {"id": 1, "start": 8.0, "end": 12.0, "text": "Intro"},
            {"id": 2, "start": 12.0, "end": 18.0, "text": "Body"},
            {"id": 3, "start": 19.0, "end": 22.0, "text": "Tail"},
            {"id": 4, "start": 25.0, "end": 28.0, "text": "Outside"},
        ]
    }

    clip_segments = build_clip_transcript_segments(
        transcript,
        clip_start_seconds=10.0,
        clip_end_seconds=20.0,
    )

    assert clip_segments == [
        {"id": 1, "start": 0.0, "end": 2.0, "text": "Intro"},
        {"id": 2, "start": 2.0, "end": 8.0, "text": "Body"},
        {"id": 3, "start": 9.0, "end": 10.0, "text": "Tail"},
    ]


def test_export_clip_subtitles_writes_rebased_srt_and_vtt(tmp_path: Path):
    transcript = {
        "segments": [
            {"id": 1, "start": 8.0, "end": 12.0, "text": "Intro"},
            {"id": 2, "start": 12.0, "end": 18.0, "text": "Body"},
        ]
    }
    clip = {
        "clip_id": "clip_001",
        "start": "00:00:10.000",
        "end": "00:00:20.000",
    }

    export_clip_subtitles(clip, transcript, tmp_path)

    assert (tmp_path / "clip_001.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:02,000\nIntro\n\n"
        "2\n00:00:02,000 --> 00:00:08,000\nBody\n"
    )
    assert (tmp_path / "clip_001.vtt").read_text(encoding="utf-8") == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\nIntro\n\n"
        "00:00:02.000 --> 00:00:08.000\nBody\n"
    )

from pathlib import Path

from scripts.cut_clips import build_cut_command, cut_clips


def test_build_cut_command_uses_accurate_transcode_defaults():
    command = build_cut_command(
        Path("/tmp/input.mp4"),
        "00:00:10.000",
        "00:00:20.000",
        Path("/tmp/output.mp4"),
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-ss",
        "00:00:10.000",
        "-to",
        "00:00:20.000",
        "-i",
        "/tmp/input.mp4",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "/tmp/output.mp4",
    ]


def test_cut_clips_runs_each_clip_and_returns_counts(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run_command(command: list[str]):
        calls.append(command)
        Path(command[-1]).write_bytes(b"clip")

    monkeypatch.setattr("scripts.cut_clips.run_command", fake_run_command)

    clips = [
        {
            "clip_id": "clip_001",
            "start": "00:00:10.000",
            "end": "00:00:20.000",
            "output_file": "output/clips/clip_001.mp4",
        },
        {
            "clip_id": "clip_002",
            "start": "00:00:21.000",
            "end": "00:00:30.000",
            "output_file": "output/clips/clip_002.mp4",
        },
    ]

    summary = cut_clips(Path("/tmp/input.mp4"), clips, tmp_path / "output" / "clips")

    assert summary == {"generated": 2, "success": 2, "failed": 0}
    assert [Path(command[-1]).name for command in calls] == ["clip_001.mp4", "clip_002.mp4"]


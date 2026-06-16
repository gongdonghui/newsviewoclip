from pathlib import Path
import subprocess

import pytest

from scripts.extract_audio import build_extract_audio_command, extract_audio
from scripts.probe_video import normalize_probe_payload


def test_normalize_probe_payload_extracts_expected_fields():
    payload = {
        "format": {"duration": "125.5", "size": "2048"},
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001",
                "codec_name": "h264",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }

    normalized = normalize_probe_payload(Path("/tmp/video.mp4"), payload)

    assert normalized == {
        "source_video": "/tmp/video.mp4",
        "duration_seconds": 125.5,
        "file_size_bytes": 2048,
        "resolution": "1920x1080",
        "frame_rate": "30000/1001",
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_sample_rate": 48000,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "format": {"size": "2048"},
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30000/1001",
                        "codec_name": "h264",
                    },
                    {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
                ],
            },
            "Missing required ffprobe field: format.duration",
        ),
    ],
)
def test_normalize_probe_payload_raises_clear_error_for_missing_requirements(
    payload: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        normalize_probe_payload(Path("/tmp/video.mp4"), payload)


def test_normalize_probe_payload_supports_video_only_input():
    payload = {
        "format": {"duration": "125.5", "size": "2048"},
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001",
                "codec_name": "h264",
            }
        ],
    }

    normalized = normalize_probe_payload(Path("/tmp/video.mp4"), payload)

    assert normalized["audio_codec"] is None
    assert normalized["audio_sample_rate"] is None


def test_build_extract_audio_command_targets_wav_output():
    command = build_extract_audio_command(
        Path("input.mp4"),
        Path("temp/source_audio.wav"),
    )

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


def test_extract_audio_falls_back_to_silent_wav_when_video_has_no_audio(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "temp" / "source_audio.wav"
    calls: list[list[str]] = []

    def fake_run_command(command: list[str]):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr="Output file does not contain any stream",
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("scripts.extract_audio.run_command", fake_run_command)

    result = extract_audio(Path("input.mp4"), output, False, duration_seconds=12.5)

    assert result == output
    assert output.exists()
    assert calls[1] == [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=16000",
        "-t",
        "12.500",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]


def test_extract_audio_reraises_non_audio_ffmpeg_failures(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "temp" / "source_audio.wav"

    def fake_run_command(command: list[str]):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="Permission denied",
        )

    monkeypatch.setattr("scripts.extract_audio.run_command", fake_run_command)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        extract_audio(Path("input.mp4"), output, False, duration_seconds=12.5)

    assert exc_info.value.stderr == "Permission denied"


def test_extract_audio_reuses_existing_output_only_for_same_source(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "temp" / "source_audio.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"wav")
    metadata_path = output.with_suffix(f"{output.suffix}.metadata.json")
    metadata_path.write_text(
        '{"source_video": "%s", "duration_seconds": 12.5}' % source,
        encoding="utf-8",
    )

    def unexpected_run(_command: list[str]):
        raise AssertionError("run_command should not be called for safe reuse")

    monkeypatch.setattr("scripts.extract_audio.run_command", unexpected_run)

    result = extract_audio(source, output, False, duration_seconds=12.5)

    assert result == output


def test_extract_audio_ignores_existing_output_when_source_changes(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "new-input.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "temp" / "source_audio.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"stale-wav")
    metadata_path = output.with_suffix(f"{output.suffix}.metadata.json")
    metadata_path.write_text(
        '{"source_video": "%s", "duration_seconds": 9.0}' % (tmp_path / "old-input.mp4"),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run_command(command: list[str]):
        calls.append(command)
        output.write_bytes(b"fresh-wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("scripts.extract_audio.run_command", fake_run_command)

    result = extract_audio(source, output, False, duration_seconds=12.5)

    assert result == output
    assert calls == [["ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)]]

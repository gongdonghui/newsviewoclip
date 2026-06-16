from pathlib import Path
import subprocess

from scripts.common import read_json
from scripts.detect_pauses import (
    DEFAULT_MIN_SILENCE_SECONDS,
    DEFAULT_NOISE_THRESHOLD_DB,
    build_pause_payload,
    detect_pauses,
    parse_silencedetect_output,
)


def test_parse_silencedetect_output_extracts_complete_pauses():
    stderr = """
[silencedetect @ 0x1] silence_start: 12.480
[silencedetect @ 0x1] silence_end: 13.210 | silence_duration: 0.730
[silencedetect @ 0x1] silence_start: 28.040
[silencedetect @ 0x1] silence_end: 28.820 | silence_duration: 0.780
""".strip()

    pauses = parse_silencedetect_output(stderr, min_silence_seconds=0.45)

    assert pauses == [
        {"start": 12.48, "end": 13.21, "duration": 0.73},
        {"start": 28.04, "end": 28.82, "duration": 0.78},
    ]


def test_parse_silencedetect_output_filters_short_pauses():
    stderr = """
[silencedetect @ 0x1] silence_start: 1.000
[silencedetect @ 0x1] silence_end: 1.300 | silence_duration: 0.300
[silencedetect @ 0x1] silence_start: 2.000
[silencedetect @ 0x1] silence_end: 2.600 | silence_duration: 0.600
""".strip()

    pauses = parse_silencedetect_output(stderr, min_silence_seconds=0.45)

    assert pauses == [{"start": 2.0, "end": 2.6, "duration": 0.6}]


def test_build_pause_payload_matches_expected_schema():
    payload = build_pause_payload(
        Path("temp/source_audio.wav"),
        noise_threshold_db=-35,
        min_silence_seconds=0.45,
        pauses=[{"start": 12.48, "end": 13.21, "duration": 0.73}],
    )

    assert payload == {
        "source_audio": "temp/source_audio.wav",
        "noise_threshold_db": -35,
        "min_silence_seconds": 0.45,
        "pauses": [{"start": 12.48, "end": 13.21, "duration": 0.73}],
    }


def test_detect_pauses_reuses_existing_output_when_not_forced(
    tmp_path: Path,
    monkeypatch,
):
    audio_path = tmp_path / "temp" / "source_audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"wav")
    output_path = tmp_path / "output" / "metadata" / "pauses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_pause_payload(
            audio_path,
            DEFAULT_NOISE_THRESHOLD_DB,
            DEFAULT_MIN_SILENCE_SECONDS,
            [{"start": 1.0, "end": 1.6, "duration": 0.6}],
        ).__str__().replace("'", '"'),
        encoding="utf-8",
    )

    def unexpected_run(_command: list[str]):
        raise AssertionError("run_command should not be called when reusing pauses.json")

    monkeypatch.setattr("scripts.detect_pauses.run_command", unexpected_run)

    payload = detect_pauses(audio_path, output_path, force=False)

    assert payload == read_json(output_path)


def test_detect_pauses_ignores_stale_cached_output_and_reruns_ffmpeg(
    tmp_path: Path,
    monkeypatch,
):
    audio_path = tmp_path / "temp" / "source_audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"wav")
    output_path = tmp_path / "output" / "metadata" / "pauses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calls: list[list[str]] = []

    def fake_run_command(command: list[str]):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            "",
            "[silencedetect @ 0x1] silence_start: 2.000\n"
            "[silencedetect @ 0x1] silence_end: 2.600 | silence_duration: 0.600",
        )

    monkeypatch.setattr("scripts.detect_pauses.run_command", fake_run_command)

    output_path.write_text(
        build_pause_payload(
            tmp_path / "temp" / "other_audio.wav",
            DEFAULT_NOISE_THRESHOLD_DB,
            DEFAULT_MIN_SILENCE_SECONDS,
            [{"start": 1.0, "end": 1.6, "duration": 0.6}],
        ).__str__().replace("'", '"'),
        encoding="utf-8",
    )

    payload = detect_pauses(audio_path, output_path, force=False)

    assert len(calls) == 1
    assert payload["source_audio"] == str(audio_path)

    calls.clear()
    output_path.write_text(
        build_pause_payload(
            audio_path,
            DEFAULT_NOISE_THRESHOLD_DB,
            DEFAULT_MIN_SILENCE_SECONDS,
            [{"start": 1.0, "end": 1.6, "duration": 0.6}],
        ).__str__().replace("'", '"'),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.detect_pauses.DEFAULT_NOISE_THRESHOLD_DB", -30)

    payload = detect_pauses(audio_path, output_path, force=False)

    assert len(calls) == 1
    assert payload["noise_threshold_db"] == -30


def test_detect_pauses_runs_ffmpeg_and_writes_payload(
    tmp_path: Path,
    monkeypatch,
):
    audio_path = tmp_path / "temp" / "source_audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"wav")
    output_path = tmp_path / "output" / "metadata" / "pauses.json"
    calls: list[list[str]] = []

    def fake_run_command(command: list[str]):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            "",
            """
[silencedetect @ 0x1] silence_start: 12.480
[silencedetect @ 0x1] silence_end: 13.210 | silence_duration: 0.730
[silencedetect @ 0x1] silence_start: 15.000
[silencedetect @ 0x1] silence_end: 15.200 | silence_duration: 0.200
""".strip(),
        )

    monkeypatch.setattr("scripts.detect_pauses.run_command", fake_run_command)

    payload = detect_pauses(audio_path, output_path, force=True)

    assert calls == [[
        "ffmpeg",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={DEFAULT_NOISE_THRESHOLD_DB}dB:d={DEFAULT_MIN_SILENCE_SECONDS}",
        "-f",
        "null",
        "-",
    ]]
    assert payload == {
        "source_audio": str(audio_path),
        "noise_threshold_db": DEFAULT_NOISE_THRESHOLD_DB,
        "min_silence_seconds": DEFAULT_MIN_SILENCE_SECONDS,
        "pauses": [{"start": 12.48, "end": 13.21, "duration": 0.73}],
    }
    assert read_json(output_path) == payload

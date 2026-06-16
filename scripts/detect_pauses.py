from __future__ import annotations

from pathlib import Path
import re

from scripts.common import read_json, run_command, write_json

DEFAULT_NOISE_THRESHOLD_DB = -35
DEFAULT_MIN_SILENCE_SECONDS = 0.45

SILENCE_START_PATTERN = re.compile(r"silence_start:\s*(?P<start>\d+(?:\.\d+)?)")
SILENCE_END_PATTERN = re.compile(
    r"silence_end:\s*(?P<end>\d+(?:\.\d+)?)\s*\|\s*silence_duration:\s*(?P<duration>\d+(?:\.\d+)?)"
)


def parse_silencedetect_output(
    stderr: str,
    min_silence_seconds: float,
) -> list[dict[str, float]]:
    pauses: list[dict[str, float]] = []
    current_start: float | None = None

    for line in stderr.splitlines():
        start_match = SILENCE_START_PATTERN.search(line)
        if start_match:
            current_start = float(start_match.group("start"))
            continue

        end_match = SILENCE_END_PATTERN.search(line)
        if not end_match or current_start is None:
            continue

        end = float(end_match.group("end"))
        duration = float(end_match.group("duration"))
        if duration >= min_silence_seconds:
            pauses.append(
                {
                    "start": current_start,
                    "end": end,
                    "duration": duration,
                }
            )
        current_start = None

    return pauses


def build_pause_payload(
    source_audio: Path,
    noise_threshold_db: int,
    min_silence_seconds: float,
    pauses: list[dict[str, float]],
) -> dict[str, object]:
    return {
        "source_audio": str(source_audio),
        "noise_threshold_db": noise_threshold_db,
        "min_silence_seconds": min_silence_seconds,
        "pauses": pauses,
    }


def can_reuse_pause_payload(
    audio_path: Path,
    output_path: Path,
    noise_threshold_db: int,
    min_silence_seconds: float,
) -> bool:
    if not output_path.exists():
        return False

    payload = read_json(output_path)
    return (
        payload.get("source_audio") == str(audio_path)
        and payload.get("noise_threshold_db") == noise_threshold_db
        and payload.get("min_silence_seconds") == min_silence_seconds
    )


def detect_pauses(
    audio_path: Path,
    output_path: Path,
    force: bool = False,
) -> dict[str, object]:
    if not force and can_reuse_pause_payload(
        audio_path,
        output_path,
        DEFAULT_NOISE_THRESHOLD_DB,
        DEFAULT_MIN_SILENCE_SECONDS,
    ):
        return read_json(output_path)

    result = run_command(
        [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-af",
            f"silencedetect=noise={DEFAULT_NOISE_THRESHOLD_DB}dB:d={DEFAULT_MIN_SILENCE_SECONDS}",
            "-f",
            "null",
            "-",
        ]
    )
    pauses = parse_silencedetect_output(result.stderr, DEFAULT_MIN_SILENCE_SECONDS)
    payload = build_pause_payload(
        audio_path,
        DEFAULT_NOISE_THRESHOLD_DB,
        DEFAULT_MIN_SILENCE_SECONDS,
        pauses,
    )
    write_json(output_path, payload)
    return payload

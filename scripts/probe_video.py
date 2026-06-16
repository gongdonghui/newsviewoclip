import json
from pathlib import Path

from scripts.common import read_json, run_command, write_json


def normalize_probe_payload(source: Path, payload: dict[str, object]) -> dict[str, object]:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError("Missing required ffprobe field: streams")

    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError("Missing required ffprobe video stream")

    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)

    format_payload = payload.get("format")
    if not isinstance(format_payload, dict):
        raise ValueError("Missing required ffprobe field: format")

    duration = format_payload.get("duration")
    if duration is None:
        raise ValueError("Missing required ffprobe field: format.duration")

    size = format_payload.get("size")
    if size is None:
        raise ValueError("Missing required ffprobe field: format.size")

    return {
        "source_video": str(source),
        "duration_seconds": float(duration),
        "file_size_bytes": int(size),
        "resolution": f"{video_stream['width']}x{video_stream['height']}",
        "frame_rate": video_stream["r_frame_rate"],
        "video_codec": video_stream["codec_name"],
        "audio_codec": audio_stream["codec_name"] if audio_stream is not None else None,
        "audio_sample_rate": (
            int(audio_stream["sample_rate"]) if audio_stream is not None else None
        ),
    }


def build_probe_command(source: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]


def probe_video(source: Path, output: Path, force: bool) -> dict[str, object]:
    if output.exists() and not force:
        payload = read_json(output)
        if payload.get("source_video") == str(source):
            return payload

    result = run_command(build_probe_command(source))
    normalized = normalize_probe_payload(source, json.loads(result.stdout))
    write_json(output, normalized)
    return normalized

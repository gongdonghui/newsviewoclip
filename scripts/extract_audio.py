from pathlib import Path
import subprocess

from scripts.common import read_json, run_command, write_json

NO_AUDIO_FAILURE_MARKERS = (
    "output file does not contain any stream",
    "stream map 'a' matches no streams",
)


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


def build_silent_audio_command(output: Path, duration_seconds: float) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=16000",
        "-t",
        f"{duration_seconds:.3f}",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]


def build_audio_metadata(source: Path, duration_seconds: float | None) -> dict[str, object]:
    metadata: dict[str, object] = {"source_video": str(source)}
    if duration_seconds is not None:
        metadata["duration_seconds"] = duration_seconds
    return metadata


def get_audio_metadata_path(output: Path) -> Path:
    return output.with_suffix(f"{output.suffix}.metadata.json")


def can_reuse_audio(
    source: Path,
    output: Path,
    duration_seconds: float | None,
) -> bool:
    metadata_path = get_audio_metadata_path(output)
    if not output.exists() or not metadata_path.exists():
        return False

    metadata = read_json(metadata_path)
    if metadata.get("source_video") != str(source):
        return False
    if duration_seconds is not None and metadata.get("duration_seconds") not in (None, duration_seconds):
        return False
    return True


def extract_audio(
    source: Path,
    output: Path,
    force: bool,
    duration_seconds: float | None = None,
) -> Path:
    metadata_path = get_audio_metadata_path(output)
    if not force and can_reuse_audio(source, output, duration_seconds):
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    command = build_extract_audio_command(source, output)
    try:
        run_command(["ffmpeg", "-y", *command[1:]])
    except subprocess.CalledProcessError as exc:
        if duration_seconds is None:
            raise
        stderr = (exc.stderr or "").lower()
        if not any(marker in stderr for marker in NO_AUDIO_FAILURE_MARKERS):
            raise
        run_command(build_silent_audio_command(output, duration_seconds))
    write_json(metadata_path, build_audio_metadata(source, duration_seconds))
    return output

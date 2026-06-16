from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from scripts.common import run_command


def build_cut_command(
    source_video: Path,
    start: str,
    end: str,
    output_path: Path,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        start,
        "-to",
        end,
        "-i",
        str(source_video),
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
        str(output_path),
    ]


def cut_clip(source_video: Path, clip: dict[str, object], clips_dir: Path) -> bool:
    output_path = clips_dir / f"{clip['clip_id']}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_cut_command(
        source_video,
        str(clip["start"]),
        str(clip["end"]),
        output_path,
    )
    try:
        run_command(command)
    except subprocess.CalledProcessError as exc:
        logging.error("Failed to cut %s: %s", clip["clip_id"], exc.stderr or exc)
        return False
    return True


def cut_clips(
    source_video: Path,
    clips: list[dict[str, object]],
    clips_dir: Path,
) -> dict[str, int]:
    success = 0
    failed = 0
    for clip in clips:
        if cut_clip(source_video, clip, clips_dir):
            success += 1
        else:
            failed += 1
    return {"generated": len(clips), "success": success, "failed": failed}

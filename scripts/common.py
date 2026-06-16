from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        values[key.strip()] = raw_value.strip().strip("\"'")
    return values


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_runtime_env(project_root: Path | None = None) -> dict[str, str]:
    root = project_root or PROJECT_ROOT
    values = load_env_file(root / ".env")
    for key, value in os.environ.items():
        if key.startswith("DEEPSEEK_"):
            values[key] = value
    return values


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_clip_plan_csv(path: Path, clips: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "title",
        "start",
        "end",
        "duration_seconds",
        "summary",
        "keywords",
        "reason",
        "confidence",
        "output_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for clip in clips:
            row = dict(clip)
            row["keywords"] = "|".join(str(item) for item in clip.get("keywords", []))
            writer.writerow(row)


def write_markdown_clip_plan(path: Path, clips: list[dict[str, object]]) -> None:
    def normalize_cell(value: object) -> str:
        return " ".join(str(value).splitlines()).replace("|", "\\|")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| 编号 | 标题 | 开始时间 | 结束时间 | 时长 | 摘要 | 建议理由 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for clip in clips:
        lines.append(
            "| {clip_id} | {title} | {start} | {end} | {duration_seconds:.1f} 秒 | {summary} | {reason} |".format(
                clip_id=clip["clip_id"],
                title=normalize_cell(clip["title"]),
                start=clip["start"],
                end=clip["end"],
                duration_seconds=clip["duration_seconds"],
                summary=normalize_cell(clip["summary"]),
                reason=normalize_cell(clip["reason"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)

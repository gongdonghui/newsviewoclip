import json
from pathlib import Path

from scripts.common import write_json, write_markdown_clip_plan


def test_write_json_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "data.json"

    write_json(path, {"language": "en"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"language": "en"}


def test_write_markdown_clip_plan_renders_table(tmp_path: Path):
    path = tmp_path / "clip_plan.md"
    clips = [
        {
            "clip_id": "clip_001",
            "title": "Opening headline",
            "start": "00:00:00.000",
            "end": "00:01:00.000",
            "duration_seconds": 60.0,
            "summary": "Anchor introduces the lead story.",
            "reason": "Self-contained introduction.",
        }
    ]

    write_markdown_clip_plan(path, clips)

    assert path.read_text(encoding="utf-8") == (
        "| 编号 | 标题 | 开始时间 | 结束时间 | 时长 | 摘要 | 建议理由 |\n"
        "|---|---|---:|---:|---:|---|---|\n"
        "| clip_001 | Opening headline | 00:00:00.000 | 00:01:00.000 | 60.0 秒 | Anchor introduces the lead story. | Self-contained introduction. |\n"
    )


def test_write_markdown_clip_plan_escapes_pipes_and_flattens_newlines(tmp_path: Path):
    path = tmp_path / "clip_plan.md"
    clips = [
        {
            "clip_id": "clip_002",
            "title": "Market | update\nnow",
            "start": "00:01:00.000",
            "end": "00:02:00.000",
            "duration_seconds": 60.04,
            "summary": "Line one\nLine | two",
            "reason": "Reason with | pipe\nand break",
        }
    ]

    write_markdown_clip_plan(path, clips)

    assert path.read_text(encoding="utf-8") == (
        "| 编号 | 标题 | 开始时间 | 结束时间 | 时长 | 摘要 | 建议理由 |\n"
        "|---|---|---:|---:|---:|---|---|\n"
        "| clip_002 | Market \\| update now | 00:01:00.000 | 00:02:00.000 | 60.0 秒 | Line one Line \\| two | Reason with \\| pipe and break |\n"
    )

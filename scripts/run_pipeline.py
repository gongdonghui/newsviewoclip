from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re

from scripts.analyze_clips import request_clip_analysis
from scripts.build_clip_plan import build_fallback_clips, normalize_clips
from scripts.common import (
    get_project_root,
    load_runtime_env,
    read_json,
    write_clip_plan_csv,
    write_json,
    write_markdown_clip_plan,
)
from scripts.cut_clips import cut_clip
from scripts.detect_pauses import detect_pauses
from scripts.extract_audio import extract_audio
from scripts.export_clip_subtitles import export_clip_subtitles
from scripts.probe_video import probe_video
from scripts.transcribe import (
    WHISPER_MODEL,
    load_or_create_empty_transcript,
    transcribe_audio,
)

MIN_CLIP_SECONDS = 15
TARGET_CLIP_SECONDS = 30
MAX_CLIP_SECONDS = 45
PADDING_BEFORE_SECONDS = 0.5
PADDING_AFTER_SECONDS = 1.0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", default="news")
    parser.add_argument("--language", default="en")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def build_plan_preview(
    transcript: dict[str, object],
    duration_seconds: float,
    min_clip_seconds: int,
    target_clip_seconds: int,
    max_clip_seconds: int,
    padding_before: float,
    padding_after: float,
    output_file_prefix: str = "output/clips",
    pauses: list[dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    raw_clips = build_fallback_clips(
        transcript,
        min_clip_seconds=min_clip_seconds,
        target_clip_seconds=target_clip_seconds,
        max_clip_seconds=max_clip_seconds,
        pauses=pauses,
    )
    return normalize_clips(
        raw_clips,
        duration_seconds,
        padding_before,
        padding_after,
        output_file_prefix=output_file_prefix,
        pauses=pauses,
        min_clip_seconds=min_clip_seconds,
        max_clip_seconds=max_clip_seconds,
    )


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock_pid(lock_path: Path) -> int | None:
    try:
        raw_value = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


@contextmanager
def acquire_pipeline_lock(output_dir: Path):
    lock_path = output_dir / ".pipeline.lock"
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing_pid = read_lock_pid(lock_path)
            if existing_pid is not None and pid_is_running(existing_pid):
                raise RuntimeError(
                    f"Pipeline for {output_dir} is already running (pid {existing_pid})"
                )
            lock_path.unlink(missing_ok=True)
            continue
        break

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))

    try:
        yield lock_path
    finally:
        lock_path.unlink(missing_ok=True)


def build_run_id(input_path: Path) -> str:
    sanitized_stem = re.sub(r"[^A-Za-z0-9]+", "-", input_path.stem).strip("-").lower()
    if not sanitized_stem:
        sanitized_stem = "video"
    digest = hashlib.sha256(str(input_path).encode("utf-8")).hexdigest()[:8]
    return f"{sanitized_stem}-{digest}"


def ensure_project_directories(project_root: Path, input_path: Path) -> dict[str, Path]:
    run_id = build_run_id(input_path)
    output_dir = project_root / "output" / run_id
    directories = {
        "output": output_dir,
        "clips": output_dir / "clips",
        "subtitles": output_dir / "subtitles",
        "metadata": output_dir / "metadata",
        "logs": output_dir / "logs",
        "temp": project_root / "temp" / run_id,
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def build_transcript_signature(transcript_payload: dict[str, object]) -> str:
    payload = {
        "source_video": transcript_payload.get("source_video"),
        "language": transcript_payload.get("language"),
        "requested_language": transcript_payload.get("requested_language"),
        "model": transcript_payload.get("model"),
        "segments": transcript_payload.get("segments", []),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_pause_signature(pauses_payload: dict[str, object] | None) -> str | None:
    if pauses_payload is None:
        return None
    payload = {
        "source_audio": pauses_payload.get("source_audio"),
        "noise_threshold_db": pauses_payload.get("noise_threshold_db"),
        "min_silence_seconds": pauses_payload.get("min_silence_seconds"),
        "pauses": pauses_payload.get("pauses", []),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_clip_plan_context(
    probe_payload: dict[str, object],
    transcript_payload: dict[str, object],
    pauses_payload: dict[str, object] | None,
    mode: str,
) -> dict[str, object]:
    return {
        "source_video": probe_payload["source_video"],
        "mode": mode,
        "transcript_signature": build_transcript_signature(transcript_payload),
        "pause_signature": build_pause_signature(pauses_payload),
    }


def load_or_create_clip_plan(
    probe_payload: dict[str, object],
    transcript_payload: dict[str, object],
    clip_plan_json: Path,
    clip_plan_csv: Path,
    clip_plan_md: Path,
    force: bool,
    env: dict[str, str],
    mode: str,
    output_file_prefix: str,
    pauses_payload: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], str]:
    context = build_clip_plan_context(probe_payload, transcript_payload, pauses_payload, mode)
    pauses = list(pauses_payload.get("pauses", [])) if pauses_payload is not None else None
    if clip_plan_json.exists() and not force:
        payload = read_json(clip_plan_json)
        clips = payload.get("clips", [])
        if payload.get("context") == context and isinstance(clips, list):
            if not clip_plan_csv.exists():
                write_clip_plan_csv(clip_plan_csv, clips)
            if not clip_plan_md.exists():
                write_markdown_clip_plan(clip_plan_md, clips)
            return clips, "reused"

    if not transcript_payload["segments"]:
        clips: list[dict[str, object]] = []
        write_json(clip_plan_json, {"context": context, "clips": clips})
        write_clip_plan_csv(clip_plan_csv, clips)
        write_markdown_clip_plan(clip_plan_md, clips)
        return clips, "fallback"

    try:
        raw_clips = request_clip_analysis(
            probe_payload,
            transcript_payload["segments"],
            env=env,
        )
        clips = normalize_clips(
            raw_clips,
            float(probe_payload["duration_seconds"]),
            PADDING_BEFORE_SECONDS,
            PADDING_AFTER_SECONDS,
            output_file_prefix=output_file_prefix,
            pauses=pauses,
            min_clip_seconds=MIN_CLIP_SECONDS,
            max_clip_seconds=MAX_CLIP_SECONDS,
        )
        if not clips:
            raise ValueError("No valid clips returned from analysis")
        analysis_source = "deepseek"
    except Exception as exc:
        logging.warning("Falling back to local clip grouping: %s", exc)
        clips = build_plan_preview(
            transcript_payload,
            float(probe_payload["duration_seconds"]),
            MIN_CLIP_SECONDS,
            TARGET_CLIP_SECONDS,
            MAX_CLIP_SECONDS,
            PADDING_BEFORE_SECONDS,
            PADDING_AFTER_SECONDS,
            output_file_prefix=output_file_prefix,
            pauses=pauses,
        )
        analysis_source = "fallback"

    write_json(clip_plan_json, {"context": context, "clips": clips})
    write_clip_plan_csv(clip_plan_csv, clips)
    write_markdown_clip_plan(clip_plan_md, clips)
    return clips, analysis_source


def print_execution_summary(
    input_path: Path,
    output_dir: Path,
    subtitles_dir: Path,
    log_path: Path,
    probe_payload: dict[str, object],
    clip_plan_md: Path,
    mode: str,
    plan_only: bool,
    execution_summary: dict[str, int],
) -> None:
    print("第一步：视频概况")
    print(f"视频路径：{input_path}")
    print(f"视频时长：{probe_payload['duration_seconds']:.1f} 秒")
    print(f"分辨率：{probe_payload['resolution']}")
    if probe_payload["audio_codec"] is None:
        print("音轨情况：未检测到音轨（已生成静音 WAV 以保持流程可运行）")
    else:
        print(
            "音轨情况："
            f"{probe_payload['audio_codec']} / {probe_payload['audio_sample_rate']} Hz"
        )
    print(f"预计输出目录：{output_dir}")
    print()
    print("第二步：切分策略")
    print(f"视频类型：{mode}")
    print("切分依据：英文新闻语义分段，DeepSeek 优先，失败时回退本地启发式分组")
    print("建议片段时长：15-45 秒，优先贴合主播单条播报或简短导语段落")
    print("是否需要字幕：是")
    print(f"是否仅生成剪辑计划：{'是' if plan_only else '否'}")
    print()
    print("第三步：候选片段")
    print(clip_plan_md.read_text(encoding="utf-8").rstrip())
    print()
    print("第四步：执行结果")
    print(f"已生成片段数量：{execution_summary['generated']}")
    print(f"成功数量：{execution_summary['success']}")
    print(f"失败数量：{execution_summary['failed']}")
    print(f"输出目录：{output_dir}")
    print(f"字幕目录：{subtitles_dir}")
    print("缩略图目录：未生成")
    print(f"日志路径：{log_path}")


def execute_clip_exports(
    source_video: Path,
    clip_plan: list[dict[str, object]],
    transcript: dict[str, object],
    clips_dir: Path,
    subtitles_dir: Path,
) -> dict[str, int]:
    success = 0
    failed = 0
    for clip in clip_plan:
        if cut_clip(source_video, clip, clips_dir):
            export_clip_subtitles(clip, transcript, subtitles_dir)
            success += 1
        else:
            failed += 1
    return {"generated": len(clip_plan), "success": success, "failed": failed}


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    project_root = get_project_root()
    directories = ensure_project_directories(project_root, input_path)
    with acquire_pipeline_lock(directories["output"]):
        log_path = directories["logs"] / "pipeline.log"
        configure_logging(log_path)
        env = load_runtime_env(project_root)
        clip_output_prefix = str(directories["clips"].relative_to(project_root))

        probe_payload = probe_video(
            input_path,
            directories["metadata"] / "source_probe.json",
            args.force,
        )
        audio_path = extract_audio(
            input_path,
            directories["temp"] / "source_audio.wav",
            args.force,
            float(probe_payload["duration_seconds"]),
        )
        pauses_payload = detect_pauses(
            audio_path,
            directories["metadata"] / "pauses.json",
            args.force,
        )
        transcript_json = directories["metadata"] / "transcript.json"
        transcript_txt = directories["metadata"] / "transcript.txt"
        transcript_srt = directories["subtitles"] / "source.srt"
        transcript_vtt = directories["subtitles"] / "source.vtt"
        if probe_payload["audio_codec"] is None:
            transcript_payload, transcript_source = load_or_create_empty_transcript(
                input_path,
                args.language,
                transcript_json,
                transcript_txt,
                transcript_srt,
                transcript_vtt,
                args.force,
                model_name=WHISPER_MODEL,
            )
        else:
            transcript_payload = transcribe_audio(
                audio_path,
                input_path,
                args.language,
                transcript_json,
                transcript_txt,
                transcript_srt,
                transcript_vtt,
                args.force,
            )
            transcript_source = "transcribed"
        clips, analysis_source = load_or_create_clip_plan(
            probe_payload,
            transcript_payload,
            directories["metadata"] / "clip_plan.json",
            directories["metadata"] / "clip_plan.csv",
            directories["metadata"] / "clip_plan.md",
            args.force,
            env,
            args.mode,
            clip_output_prefix,
            pauses_payload,
        )

        if args.execute:
            execution_summary = execute_clip_exports(
                input_path,
                clips,
                transcript_payload,
                directories["clips"],
                directories["subtitles"],
            )
        else:
            execution_summary = {"generated": 0, "success": 0, "failed": 0}

    print_execution_summary(
        input_path,
        directories["output"],
        directories["subtitles"],
        log_path,
        probe_payload,
        directories["metadata"] / "clip_plan.md",
        args.mode,
        not args.execute,
        execution_summary,
    )

    return {
        "probe": probe_payload,
        "audio_path": audio_path,
        "transcript": transcript_payload,
        "pauses": pauses_payload,
        "transcript_source": transcript_source,
        "clips": clips,
        "analysis_source": analysis_source,
        "execution_summary": execution_summary,
        "log_path": log_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

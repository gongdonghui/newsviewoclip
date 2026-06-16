import json
import logging
import os
from pathlib import Path

import pytest

from scripts.run_pipeline import (
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    TARGET_CLIP_SECONDS,
    build_argument_parser,
    build_run_id,
    build_plan_preview,
    build_clip_plan_context,
    load_or_create_clip_plan,
)


def test_argument_parser_supports_plan_only():
    parser = build_argument_parser()
    args = parser.parse_args(
        ["--input", "/tmp/video.mp4", "--mode", "news", "--language", "en", "--plan-only"]
    )

    assert str(args.input) == "/tmp/video.mp4"
    assert args.mode == "news"
    assert args.language == "en"
    assert args.plan_only is True
    assert args.execute is False
    assert args.force is False

    forced_args = parser.parse_args(["--input", "/tmp/video.mp4", "--force"])
    assert forced_args.force is True

    execute_args = parser.parse_args(["--input", "/tmp/video.mp4", "--execute"])
    assert execute_args.execute is True


def test_fixture_transcript_produces_shorter_clip_plan():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 30.0, "text": "Opening headline."},
            {"id": 2, "start": 31.0, "end": 70.0, "text": "Reporter package."},
        ]
    }

    clips = build_plan_preview(
        transcript,
        duration_seconds=120.0,
        min_clip_seconds=MIN_CLIP_SECONDS,
        target_clip_seconds=TARGET_CLIP_SECONDS,
        max_clip_seconds=MAX_CLIP_SECONDS,
        padding_before=0.5,
        padding_after=1.0,
    )

    assert clips[0]["clip_id"] == "clip_001"
    assert clips[0]["start"] == "00:00:00.000"
    assert clips[0]["end"] == "00:00:31.000"
    assert clips[0]["duration_seconds"] == 31.0
    assert clips[1]["clip_id"] == "clip_002"
    assert clips[1]["start"] == "00:00:30.500"
    assert clips[1]["end"] == "00:01:11.000"


def test_run_pipeline_plan_only_writes_artifacts_and_falls_back_when_analysis_fails(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    run_id = build_run_id(input_path)
    run_output = tmp_path / "output" / run_id
    run_temp = tmp_path / "temp" / run_id

    probe_payload = {
        "source_video": str(input_path),
        "duration_seconds": 120.0,
        "file_size_bytes": 5,
        "resolution": "1920x1080",
        "frame_rate": "30/1",
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_sample_rate": 48000,
    }
    transcript_payload = {
        "source_video": str(input_path),
        "language": "en",
        "requested_language": "en",
        "model": "mlx-community/whisper-large-v3-mlx",
        "segments": [
            {"id": 1, "start": 0.0, "end": 35.0, "text": "Opening headline."},
            {"id": 2, "start": 36.0, "end": 70.0, "text": "Reporter details."},
        ],
    }
    pause_payload = {
        "source_audio": str(run_temp / "source_audio.wav"),
        "noise_threshold_db": -35,
        "min_silence_seconds": 0.45,
        "pauses": [{"start": 34.6, "end": 35.2, "duration": 0.6}],
    }

    def fake_probe_video(source: Path, output: Path, force: bool) -> dict[str, object]:
        assert source == input_path
        assert force is False
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(probe_payload), encoding="utf-8")
        return probe_payload

    def fake_extract_audio(
        source: Path,
        output: Path,
        force: bool,
        duration_seconds: float | None = None,
    ) -> Path:
        assert source == input_path
        assert force is False
        assert duration_seconds == 120.0
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return output

    def fake_transcribe_audio(
        audio_path: Path,
        source_video: Path,
        language: str,
        transcript_json: Path,
        transcript_txt: Path,
        transcript_srt: Path,
        transcript_vtt: Path,
        force: bool,
    ) -> dict[str, object]:
        assert audio_path == run_temp / "source_audio.wav"
        assert source_video == input_path
        assert language == "en"
        assert force is False
        transcript_json.parent.mkdir(parents=True, exist_ok=True)
        transcript_srt.parent.mkdir(parents=True, exist_ok=True)
        transcript_json.write_text(json.dumps(transcript_payload), encoding="utf-8")
        transcript_txt.write_text("Opening headline.\nReporter details.\n", encoding="utf-8")
        transcript_srt.write_text("1\n00:00:00,000 --> 00:00:35,000\nOpening headline.\n", encoding="utf-8")
        transcript_vtt.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:35.000\nOpening headline.\n",
            encoding="utf-8",
        )
        return transcript_payload

    def fake_request_clip_analysis(*_args, **_kwargs):
        raise ValueError("malformed analysis")

    def fake_detect_pauses(audio_path: Path, output_path: Path, force: bool) -> dict[str, object]:
        assert audio_path == run_temp / "source_audio.wav"
        assert force is False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(pause_payload), encoding="utf-8")
        return pause_payload

    monkeypatch.setattr("scripts.run_pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("scripts.run_pipeline.probe_video", fake_probe_video)
    monkeypatch.setattr("scripts.run_pipeline.extract_audio", fake_extract_audio)
    monkeypatch.setattr("scripts.run_pipeline.transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr("scripts.run_pipeline.request_clip_analysis", fake_request_clip_analysis)
    monkeypatch.setattr("scripts.run_pipeline.detect_pauses", fake_detect_pauses)

    from scripts.run_pipeline import run_pipeline

    result = run_pipeline(
        build_argument_parser().parse_args(
            ["--input", str(input_path), "--mode", "news", "--language", "en", "--plan-only"]
        )
    )

    assert result["analysis_source"] == "fallback"
    assert (run_output / "metadata" / "source_probe.json").exists()
    assert (run_temp / "source_audio.wav").exists()
    assert (run_output / "metadata" / "transcript.json").exists()
    assert (run_output / "metadata" / "pauses.json").exists()
    assert (run_output / "metadata" / "transcript.txt").exists()
    assert (run_output / "subtitles" / "source.srt").exists()
    assert (run_output / "subtitles" / "source.vtt").exists()
    assert (run_output / "metadata" / "clip_plan.json").exists()
    assert (run_output / "metadata" / "clip_plan.csv").exists()
    assert (run_output / "metadata" / "clip_plan.md").exists()
    assert list((run_output / "clips").glob("*.mp4")) == []
    assert result["pauses"] == pause_payload

    captured = capsys.readouterr().out
    assert "视频路径：" in captured
    assert "建议片段时长：15-45 秒，优先贴合主播单条播报或简短导语段落" in captured
    assert "是否仅生成剪辑计划：是" in captured
    assert "clip_001" in captured


def test_run_pipeline_execute_cuts_clips_and_exports_subtitles(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    run_id = build_run_id(input_path)
    run_output = tmp_path / "output" / run_id
    run_temp = tmp_path / "temp" / run_id

    probe_payload = {
        "source_video": str(input_path),
        "duration_seconds": 120.0,
        "file_size_bytes": 5,
        "resolution": "1920x1080",
        "frame_rate": "30/1",
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_sample_rate": 48000,
    }
    transcript_payload = {
        "source_video": str(input_path),
        "language": "en",
        "requested_language": "en",
        "model": "mlx-community/whisper-large-v3-mlx",
        "segments": [
            {"id": 1, "start": 9.0, "end": 16.0, "text": "Opening headline."},
            {"id": 2, "start": 16.0, "end": 22.0, "text": "Reporter details."},
        ],
    }
    clips = [
        {
            "clip_id": "clip_001",
            "title": "Lead",
            "start": "00:00:10.000",
            "end": "00:00:20.000",
            "duration_seconds": 10.0,
            "summary": "Lead story.",
            "keywords": ["lead"],
            "reason": "Complete topic.",
            "confidence": 0.9,
            "output_file": f"output/{run_id}/clips/clip_001.mp4",
        }
    ]

    monkeypatch.setattr("scripts.run_pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("scripts.run_pipeline.probe_video", lambda *_args, **_kwargs: probe_payload)
    monkeypatch.setattr(
        "scripts.run_pipeline.extract_audio",
        lambda *_args, **_kwargs: run_temp / "source_audio.wav",
    )
    monkeypatch.setattr(
        "scripts.run_pipeline.detect_pauses",
        lambda audio_path, output_path, force: {
            "source_audio": str(audio_path),
            "noise_threshold_db": -35,
            "min_silence_seconds": 0.45,
            "pauses": [],
        },
    )
    monkeypatch.setattr(
        "scripts.run_pipeline.transcribe_audio",
        lambda *_args, **_kwargs: transcript_payload,
    )
    monkeypatch.setattr(
        "scripts.run_pipeline.load_or_create_clip_plan",
        lambda *_args, **_kwargs: (clips, "deepseek"),
    )
    clip_plan_md = run_output / "metadata" / "clip_plan.md"
    clip_plan_md.parent.mkdir(parents=True, exist_ok=True)
    clip_plan_md.write_text(
        "| 编号 | 标题 | 开始时间 | 结束时间 | 时长 | 摘要 | 建议理由 |\n"
        "|---|---|---:|---:|---:|---|---|\n"
        "| clip_001 | Lead | 00:00:10.000 | 00:00:20.000 | 10.0 秒 | Lead story. | Complete topic. |\n",
        encoding="utf-8",
    )

    execution_calls: list[tuple[Path, list[dict[str, object]], dict[str, object], Path, Path]] = []

    def fake_execute_clip_exports(
        source_video: Path,
        clip_plan: list[dict[str, object]],
        transcript: dict[str, object],
        clips_dir: Path,
        subtitles_dir: Path,
    ) -> dict[str, int]:
        execution_calls.append(
            (source_video, clip_plan, transcript, clips_dir, subtitles_dir)
        )
        return {"generated": 1, "success": 1, "failed": 0}

    monkeypatch.setattr("scripts.run_pipeline.execute_clip_exports", fake_execute_clip_exports)

    from scripts.run_pipeline import run_pipeline

    result = run_pipeline(
        build_argument_parser().parse_args(
            ["--input", str(input_path), "--mode", "news", "--language", "en", "--execute"]
        )
    )

    assert result["execution_summary"] == {"generated": 1, "success": 1, "failed": 0}
    assert execution_calls == [
        (
            input_path,
            clips,
            transcript_payload,
            run_output / "clips",
            run_output / "subtitles",
        )
    ]

    captured = capsys.readouterr().out
    assert "是否仅生成剪辑计划：否" in captured
    assert "已生成片段数量：1" in captured
    assert "成功数量：1" in captured
    assert "失败数量：0" in captured


def test_run_pipeline_discards_transcript_for_video_only_source(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    run_id = build_run_id(input_path)
    run_output = tmp_path / "output" / run_id
    run_temp = tmp_path / "temp" / run_id

    probe_payload = {
        "source_video": str(input_path),
        "duration_seconds": 120.0,
        "file_size_bytes": 5,
        "resolution": "1920x1080",
        "frame_rate": "30/1",
        "video_codec": "h264",
        "audio_codec": None,
        "audio_sample_rate": None,
    }
    transcript_payload = {
        "source_video": str(input_path),
        "language": "en",
        "model": "mlx-community/whisper-large-v3-mlx",
        "segments": [
            {"id": 1, "start": 0.0, "end": 35.0, "text": "hallucinated text"},
        ],
    }

    monkeypatch.setattr("scripts.run_pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("scripts.run_pipeline.probe_video", lambda *_args, **_kwargs: probe_payload)
    monkeypatch.setattr(
        "scripts.run_pipeline.extract_audio",
        lambda *_args, **_kwargs: run_temp / "source_audio.wav",
    )
    monkeypatch.setattr(
        "scripts.run_pipeline.detect_pauses",
        lambda audio_path, output_path, force: {
            "source_audio": str(audio_path),
            "noise_threshold_db": -35,
            "min_silence_seconds": 0.45,
            "pauses": [],
        },
    )
    monkeypatch.setattr(
        "scripts.run_pipeline.transcribe_audio",
        lambda *_args, **_kwargs: transcript_payload,
    )
    monkeypatch.setattr(
        "scripts.run_pipeline.request_clip_analysis",
        lambda *_args, **_kwargs: [
            {
                "clip_id": "clip_001",
                "title": "Bad clip",
                "start": 0.0,
                "end": 60.0,
                "summary": "Should not survive video-only input.",
                "keywords": [],
                "reason": "Bad.",
                "confidence": 0.5,
            }
        ],
    )

    from scripts.run_pipeline import run_pipeline

    result = run_pipeline(
        build_argument_parser().parse_args(
            ["--input", str(input_path), "--mode", "news", "--language", "en", "--plan-only"]
        )
    )

    assert result["transcript"]["segments"] == []
    assert result["clips"] == []
    transcript_file = run_output / "metadata" / "transcript.json"
    assert json.loads(transcript_file.read_text(encoding="utf-8"))["segments"] == []


def test_run_pipeline_no_audio_path_skips_transcription_and_reuses_empty_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    run_id = build_run_id(input_path)
    run_output = tmp_path / "output" / run_id
    run_temp = tmp_path / "temp" / run_id
    probe_payload = {
        "source_video": str(input_path),
        "duration_seconds": 120.0,
        "file_size_bytes": 5,
        "resolution": "1920x1080",
        "frame_rate": "30/1",
        "video_codec": "h264",
        "audio_codec": None,
        "audio_sample_rate": None,
    }

    monkeypatch.setattr("scripts.run_pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("scripts.run_pipeline.probe_video", lambda *_args, **_kwargs: probe_payload)
    monkeypatch.setattr(
        "scripts.run_pipeline.extract_audio",
        lambda *_args, **_kwargs: run_temp / "source_audio.wav",
    )
    monkeypatch.setattr(
        "scripts.run_pipeline.detect_pauses",
        lambda audio_path, output_path, force: {
            "source_audio": str(audio_path),
            "noise_threshold_db": -35,
            "min_silence_seconds": 0.45,
            "pauses": [],
        },
    )

    def unexpected_transcribe(*_args, **_kwargs):
        raise AssertionError("transcribe_audio should not be called for no-audio inputs")

    monkeypatch.setattr("scripts.run_pipeline.transcribe_audio", unexpected_transcribe)

    from scripts.run_pipeline import run_pipeline

    result = run_pipeline(
        build_argument_parser().parse_args(
            ["--input", str(input_path), "--mode", "news", "--language", "en", "--plan-only"]
        )
    )

    assert result["analysis_source"] == "fallback"
    assert result["transcript"]["segments"] == []
    assert result["clips"] == []

    transcript_json = run_output / "metadata" / "transcript.json"
    clip_plan_json = run_output / "metadata" / "clip_plan.json"
    transcript_before = transcript_json.stat().st_mtime_ns
    clip_plan_before = clip_plan_json.stat().st_mtime_ns

    result = run_pipeline(
        build_argument_parser().parse_args(
            ["--input", str(input_path), "--mode", "news", "--language", "en", "--plan-only"]
        )
    )

    assert result["analysis_source"] == "reused"
    assert transcript_json.stat().st_mtime_ns == transcript_before
    assert clip_plan_json.stat().st_mtime_ns == clip_plan_before


def test_run_pipeline_uses_input_specific_artifact_paths(
    tmp_path: Path,
    monkeypatch,
):
    first_input = tmp_path / "first.mp4"
    second_input = tmp_path / "second.mp4"
    first_input.write_bytes(b"first")
    second_input.write_bytes(b"second")

    probe_payloads = {
        first_input: {
            "source_video": str(first_input),
            "duration_seconds": 60.0,
            "file_size_bytes": 5,
            "resolution": "1920x1080",
            "frame_rate": "30/1",
            "video_codec": "h264",
            "audio_codec": "aac",
            "audio_sample_rate": 48000,
        },
        second_input: {
            "source_video": str(second_input),
            "duration_seconds": 90.0,
            "file_size_bytes": 6,
            "resolution": "1280x720",
            "frame_rate": "30/1",
            "video_codec": "h264",
            "audio_codec": "aac",
            "audio_sample_rate": 44100,
        },
    }
    transcript_payloads = {
        first_input: {
            "source_video": str(first_input),
            "language": "en",
            "requested_language": "en",
            "model": "mlx-community/whisper-large-v3-mlx",
            "segments": [{"id": 1, "start": 0.0, "end": 20.0, "text": "First input"}],
        },
        second_input: {
            "source_video": str(second_input),
            "language": "en",
            "requested_language": "en",
            "model": "mlx-community/whisper-large-v3-mlx",
            "segments": [{"id": 1, "start": 0.0, "end": 30.0, "text": "Second input"}],
        },
    }

    def fake_probe_video(source: Path, output: Path, force: bool) -> dict[str, object]:
        payload = probe_payloads[source]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_extract_audio(
        source: Path,
        output: Path,
        force: bool,
        duration_seconds: float | None = None,
    ) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.name.encode("utf-8"))
        return output

    def fake_transcribe_audio(
        audio_path: Path,
        source_video: Path,
        language: str,
        transcript_json: Path,
        transcript_txt: Path,
        transcript_srt: Path,
        transcript_vtt: Path,
        force: bool,
    ) -> dict[str, object]:
        payload = transcript_payloads[source_video]
        transcript_json.parent.mkdir(parents=True, exist_ok=True)
        transcript_srt.parent.mkdir(parents=True, exist_ok=True)
        transcript_json.write_text(json.dumps(payload), encoding="utf-8")
        transcript_txt.write_text(f"{source_video.name}\n", encoding="utf-8")
        transcript_srt.write_text("1\n00:00:00,000 --> 00:00:20,000\ntext\n", encoding="utf-8")
        transcript_vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:20.000\ntext\n", encoding="utf-8")
        return payload

    monkeypatch.setattr("scripts.run_pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("scripts.run_pipeline.probe_video", fake_probe_video)
    monkeypatch.setattr("scripts.run_pipeline.extract_audio", fake_extract_audio)
    monkeypatch.setattr(
        "scripts.run_pipeline.detect_pauses",
        lambda audio_path, output_path, force: {
            "source_audio": str(audio_path),
            "noise_threshold_db": -35,
            "min_silence_seconds": 0.45,
            "pauses": [],
        },
    )
    monkeypatch.setattr("scripts.run_pipeline.transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(
        "scripts.run_pipeline.request_clip_analysis",
        lambda *_args, **_kwargs: [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 0.0,
                "end": 20.0,
                "summary": "Lead story.",
                "keywords": [],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
    )

    from scripts.run_pipeline import run_pipeline

    first_result = run_pipeline(
        build_argument_parser().parse_args(
            ["--input", str(first_input), "--mode", "news", "--language", "en", "--plan-only"]
        )
    )
    second_result = run_pipeline(
        build_argument_parser().parse_args(
            ["--input", str(second_input), "--mode", "news", "--language", "en", "--plan-only"]
        )
    )

    assert first_result["audio_path"] != second_result["audio_path"]
    assert first_result["log_path"] != second_result["log_path"]
    assert first_result["transcript"]["source_video"] != second_result["transcript"]["source_video"]


def test_run_pipeline_rejects_concurrent_run_for_same_input(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    run_dir = tmp_path / "output" / build_run_id(input_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".pipeline.lock").write_text(str(os.getpid()), encoding="utf-8")

    monkeypatch.setattr("scripts.run_pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "scripts.run_pipeline.probe_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("probe_video should not run when the input is already locked")
        ),
    )

    from scripts.run_pipeline import run_pipeline

    with pytest.raises(RuntimeError, match="already running"):
        run_pipeline(
            build_argument_parser().parse_args(
                ["--input", str(input_path), "--mode", "news", "--language", "en", "--plan-only"]
            )
        )


def test_load_or_create_clip_plan_reuses_only_matching_context(
    tmp_path: Path,
    monkeypatch,
):
    clip_plan_json = tmp_path / "output" / "metadata" / "clip_plan.json"
    clip_plan_csv = tmp_path / "output" / "metadata" / "clip_plan.csv"
    clip_plan_md = tmp_path / "output" / "metadata" / "clip_plan.md"
    clip_plan_json.parent.mkdir(parents=True, exist_ok=True)
    probe_payload = {
        "source_video": "/tmp/video.mp4",
        "duration_seconds": 120.0,
        "resolution": "1920x1080",
        "frame_rate": "30/1",
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_sample_rate": 48000,
    }
    transcript_payload = {
        "source_video": "/tmp/video.mp4",
        "language": "en",
        "requested_language": "en",
        "model": "mlx-community/whisper-large-v3-mlx",
        "segments": [{"id": 1, "start": 0.0, "end": 35.0, "text": "Opening headline."}],
    }
    pauses_payload = {
        "source_audio": "/tmp/source_audio.wav",
        "noise_threshold_db": -35,
        "min_silence_seconds": 0.45,
        "pauses": [{"start": 34.6, "end": 35.2, "duration": 0.6}],
    }
    expected_clips = [
        {
            "clip_id": "clip_001",
            "title": "Lead",
            "start": "00:00:00.000",
            "end": "00:00:36.000",
            "duration_seconds": 36.0,
            "summary": "Lead story.",
            "keywords": [],
            "reason": "Complete.",
            "confidence": 0.9,
            "output_file": "output/example-run/clips/clip_001.mp4",
        }
    ]

    clip_plan_json.write_text(
        json.dumps(
            {
                "context": build_clip_plan_context(
                    probe_payload,
                    transcript_payload,
                    pauses_payload,
                    "news",
                ),
                "clips": expected_clips,
            }
        ),
        encoding="utf-8",
    )

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("analysis should not rerun when clip-plan context matches")

    monkeypatch.setattr("scripts.run_pipeline.request_clip_analysis", unexpected_request)

    clips, source = load_or_create_clip_plan(
        probe_payload,
        transcript_payload,
        clip_plan_json,
        clip_plan_csv,
        clip_plan_md,
        False,
        {},
        "news",
        "output/example-run/clips",
        pauses_payload,
    )

    assert source == "reused"
    assert clips == expected_clips


def test_load_or_create_clip_plan_recomputes_when_context_changes(
    tmp_path: Path,
    monkeypatch,
):
    clip_plan_json = tmp_path / "output" / "metadata" / "clip_plan.json"
    clip_plan_csv = tmp_path / "output" / "metadata" / "clip_plan.csv"
    clip_plan_md = tmp_path / "output" / "metadata" / "clip_plan.md"
    clip_plan_json.parent.mkdir(parents=True, exist_ok=True)
    probe_payload = {
        "source_video": "/tmp/video.mp4",
        "duration_seconds": 120.0,
        "resolution": "1920x1080",
        "frame_rate": "30/1",
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_sample_rate": 48000,
    }
    transcript_payload = {
        "source_video": "/tmp/video.mp4",
        "language": "en",
        "requested_language": "en",
        "model": "mlx-community/whisper-large-v3-mlx",
        "segments": [{"id": 1, "start": 0.0, "end": 35.0, "text": "Opening headline."}],
    }
    pauses_payload = {
        "source_audio": "/tmp/source_audio.wav",
        "noise_threshold_db": -35,
        "min_silence_seconds": 0.45,
        "pauses": [{"start": 34.6, "end": 35.2, "duration": 0.6}],
    }
    clip_plan_json.write_text(
        json.dumps(
            {
                "context": {
                    "source_video": "/tmp/video.mp4",
                    "mode": "news",
                    "transcript_signature": "stale",
                    "pause_signature": "stale",
                },
                "clips": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_request(*_args, **_kwargs):
        return [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 0.0,
                "end": 35.0,
                "summary": "Lead story.",
                "keywords": [],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ]

    monkeypatch.setattr("scripts.run_pipeline.request_clip_analysis", fake_request)
    monkeypatch.setattr(logging, "warning", lambda *_args, **_kwargs: None)

    clips, source = load_or_create_clip_plan(
        probe_payload,
        transcript_payload,
        clip_plan_json,
        clip_plan_csv,
        clip_plan_md,
        False,
        {},
        "news",
        "output/example-run/clips",
        pauses_payload,
    )

    assert source == "deepseek"
    assert clips[0]["clip_id"] == "clip_001"
    persisted = json.loads(clip_plan_json.read_text(encoding="utf-8"))
    assert persisted["context"]["transcript_signature"] != "stale"


def test_load_or_create_clip_plan_recomputes_when_pause_signature_changes(
    tmp_path: Path,
    monkeypatch,
):
    clip_plan_json = tmp_path / "output" / "metadata" / "clip_plan.json"
    clip_plan_csv = tmp_path / "output" / "metadata" / "clip_plan.csv"
    clip_plan_md = tmp_path / "output" / "metadata" / "clip_plan.md"
    clip_plan_json.parent.mkdir(parents=True, exist_ok=True)
    probe_payload = {
        "source_video": "/tmp/video.mp4",
        "duration_seconds": 120.0,
        "resolution": "1920x1080",
        "frame_rate": "30/1",
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_sample_rate": 48000,
    }
    transcript_payload = {
        "source_video": "/tmp/video.mp4",
        "language": "en",
        "requested_language": "en",
        "model": "mlx-community/whisper-large-v3-mlx",
        "segments": [{"id": 1, "start": 0.0, "end": 35.0, "text": "Opening headline."}],
    }
    old_pauses_payload = {
        "source_audio": "/tmp/source_audio.wav",
        "noise_threshold_db": -35,
        "min_silence_seconds": 0.45,
        "pauses": [{"start": 30.0, "end": 30.6, "duration": 0.6}],
    }
    new_pauses_payload = {
        "source_audio": "/tmp/source_audio.wav",
        "noise_threshold_db": -35,
        "min_silence_seconds": 0.45,
        "pauses": [{"start": 34.6, "end": 35.2, "duration": 0.6}],
    }
    clip_plan_json.write_text(
        json.dumps(
            {
                "context": build_clip_plan_context(
                    probe_payload,
                    transcript_payload,
                    old_pauses_payload,
                    "news",
                ),
                "clips": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.run_pipeline.request_clip_analysis",
        lambda *_args, **_kwargs: [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 0.0,
                "end": 35.0,
                "summary": "Lead story.",
                "keywords": [],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
    )
    monkeypatch.setattr(logging, "warning", lambda *_args, **_kwargs: None)

    clips, source = load_or_create_clip_plan(
        probe_payload,
        transcript_payload,
        clip_plan_json,
        clip_plan_csv,
        clip_plan_md,
        False,
        {},
        "news",
        "output/example-run/clips",
        new_pauses_payload,
    )

    assert source == "deepseek"
    assert clips[0]["clip_id"] == "clip_001"

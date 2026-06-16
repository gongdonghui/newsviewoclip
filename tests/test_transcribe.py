import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.transcribe import (
    build_transcript_payload,
    transcribe_audio,
    render_srt_entries,
    render_vtt_entries,
)


def test_build_transcript_payload_preserves_segments():
    payload = build_transcript_payload(
        "/tmp/video.mp4",
        "en",
        "en",
        "mlx-community/whisper-large-v3-mlx",
        [(0.0, 1.25, "  Hello world.  ")],
    )

    assert payload["source_video"] == "/tmp/video.mp4"
    assert payload["language"] == "en"
    assert payload["requested_language"] == "en"
    assert payload["model"] == "mlx-community/whisper-large-v3-mlx"
    assert payload["segments"] == [
        {"id": 1, "start": 0.0, "end": 1.25, "text": "Hello world."}
    ]

def test_render_srt_and_vtt_match_expected_output():
    segments = [{"id": 1, "start": 0.0, "end": 1.25, "text": "Hello world."}]

    srt = render_srt_entries(segments)
    vtt = render_vtt_entries(segments)

    assert srt == "1\n00:00:00,000 --> 00:00:01,250\nHello world.\n"
    assert vtt == "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello world.\n"


def test_transcribe_audio_reuses_existing_transcript_when_context_matches(
    tmp_path: Path,
    monkeypatch,
):
    transcript_json = tmp_path / "output" / "metadata" / "transcript.json"
    transcript_txt = tmp_path / "output" / "metadata" / "transcript.txt"
    transcript_srt = tmp_path / "output" / "subtitles" / "source.srt"
    transcript_vtt = tmp_path / "output" / "subtitles" / "source.vtt"
    transcript_json.parent.mkdir(parents=True, exist_ok=True)
    transcript_srt.parent.mkdir(parents=True, exist_ok=True)
    transcript_json.write_text(
        """
{
  "source_video": "/tmp/video.mp4",
  "language": "en",
  "requested_language": "en",
  "model": "mlx-community/whisper-large-v3-mlx",
  "segments": [{"id": 1, "start": 0.0, "end": 1.0, "text": "Hello"}]
}
""".strip(),
        encoding="utf-8",
    )

    def unexpected_import():
        raise AssertionError("mlx_whisper should not be loaded for safe reuse")

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=unexpected_import))

    payload = transcribe_audio(
        tmp_path / "temp" / "source_audio.wav",
        Path("/tmp/video.mp4"),
        "en",
        transcript_json,
        transcript_txt,
        transcript_srt,
        transcript_vtt,
        False,
    )

    assert payload["segments"] == [{"id": 1, "start": 0.0, "end": 1.0, "text": "Hello"}]
    assert transcript_txt.read_text(encoding="utf-8") == "Hello\n"


@pytest.mark.parametrize(
    ("payload_override", "requested_language", "model_name"),
    [
        ({"source_video": "/tmp/other.mp4"}, "en", "mlx-community/whisper-large-v3-mlx"),
        ({"requested_language": "fr"}, "en", "mlx-community/whisper-large-v3-mlx"),
        ({"model": "other-model"}, "en", "mlx-community/whisper-large-v3-mlx"),
    ],
)
def test_transcribe_audio_reprocesses_when_reuse_context_changes(
    tmp_path: Path,
    monkeypatch,
    payload_override: dict[str, object],
    requested_language: str,
    model_name: str,
):
    transcript_json = tmp_path / "output" / "metadata" / "transcript.json"
    transcript_txt = tmp_path / "output" / "metadata" / "transcript.txt"
    transcript_srt = tmp_path / "output" / "subtitles" / "source.srt"
    transcript_vtt = tmp_path / "output" / "subtitles" / "source.vtt"
    transcript_json.parent.mkdir(parents=True, exist_ok=True)
    transcript_srt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_video": "/tmp/video.mp4",
        "language": "en",
        "requested_language": "en",
        "model": "mlx-community/whisper-large-v3-mlx",
        "segments": [{"id": 1, "start": 0.0, "end": 1.0, "text": "Stale"}],
    }
    payload.update(payload_override)
    transcript_json.write_text(json.dumps(payload), encoding="utf-8")

    calls: list[tuple[str, str]] = []

    def fake_transcribe(audio: str, *, path_or_hf_repo: str, **decode_options):
        calls.append((audio, path_or_hf_repo))
        assert decode_options["language"] == requested_language
        return {
            "language": requested_language,
            "segments": [{"start": 0.0, "end": 1.25, "text": "Fresh"}],
        }

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=fake_transcribe))

    fresh = transcribe_audio(
        tmp_path / "temp" / "source_audio.wav",
        Path("/tmp/video.mp4"),
        requested_language,
        transcript_json,
        transcript_txt,
        transcript_srt,
        transcript_vtt,
        False,
        model_name=model_name,
    )

    assert calls == [(str(tmp_path / "temp" / "source_audio.wav"), model_name)]
    assert fresh["segments"] == [{"id": 1, "start": 0.0, "end": 1.25, "text": "Fresh"}]

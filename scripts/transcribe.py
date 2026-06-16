from __future__ import annotations

from pathlib import Path

from scripts.common import read_json, write_json
from scripts.common import seconds_to_timestamp

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"


def build_transcript_payload(
    source_video: str,
    language: str,
    requested_language: str,
    model_name: str,
    raw_segments: list[tuple[float, float, str]],
) -> dict[str, object]:
    segments = []
    for index, (start, end, text) in enumerate(raw_segments, start=1):
        segments.append({"id": index, "start": start, "end": end, "text": text.strip()})

    return {
        "source_video": source_video,
        "language": language,
        "requested_language": requested_language,
        "model": model_name,
        "segments": segments,
    }


def build_empty_transcript_payload(
    source_video: str,
    requested_language: str,
    model_name: str,
) -> dict[str, object]:
    return {
        "source_video": source_video,
        "language": requested_language,
        "requested_language": requested_language,
        "model": model_name,
        "segments": [],
    }


def render_srt_entries(segments: list[dict[str, object]]) -> str:
    blocks = []
    for segment in segments:
        start = seconds_to_timestamp(segment["start"]).replace(".", ",")
        end = seconds_to_timestamp(segment["end"]).replace(".", ",")
        blocks.append(f"{segment['id']}\n{start} --> {end}\n{segment['text']}")

    return "\n\n".join(blocks) + "\n"


def render_vtt_entries(segments: list[dict[str, object]]) -> str:
    blocks = ["WEBVTT\n"]
    for segment in segments:
        start = seconds_to_timestamp(segment["start"])
        end = seconds_to_timestamp(segment["end"])
        blocks.append(f"{start} --> {end}\n{segment['text']}\n")

    return "\n".join(blocks)


def write_transcript_artifacts(
    payload: dict[str, object],
    transcript_json: Path,
    transcript_txt: Path,
    transcript_srt: Path,
    transcript_vtt: Path,
) -> None:
    segments = payload["segments"]
    write_json(transcript_json, payload)
    transcript_txt.parent.mkdir(parents=True, exist_ok=True)
    transcript_txt.write_text(
        "\n".join(str(segment["text"]) for segment in segments) + "\n",
        encoding="utf-8",
    )
    transcript_srt.parent.mkdir(parents=True, exist_ok=True)
    transcript_srt.write_text(render_srt_entries(segments), encoding="utf-8")
    transcript_vtt.write_text(render_vtt_entries(segments), encoding="utf-8")


def can_reuse_transcript(
    payload: dict[str, object],
    source_video: Path,
    language: str,
    model_name: str,
) -> bool:
    return (
        payload.get("source_video") == str(source_video)
        and payload.get("requested_language") == language
        and payload.get("model") == model_name
    )


def load_or_create_empty_transcript(
    source_video: Path,
    language: str,
    transcript_json: Path,
    transcript_txt: Path,
    transcript_srt: Path,
    transcript_vtt: Path,
    force: bool,
    model_name: str = WHISPER_MODEL,
) -> tuple[dict[str, object], str]:
    if transcript_json.exists() and not force:
        payload = read_json(transcript_json)
        if can_reuse_transcript(payload, source_video, language, model_name) and payload.get("segments") == []:
            if not transcript_txt.exists() or not transcript_srt.exists() or not transcript_vtt.exists():
                write_transcript_artifacts(
                    payload,
                    transcript_json,
                    transcript_txt,
                    transcript_srt,
                    transcript_vtt,
                )
            return payload, "reused"

    payload = build_empty_transcript_payload(str(source_video), language, model_name)
    write_transcript_artifacts(
        payload,
        transcript_json,
        transcript_txt,
        transcript_srt,
        transcript_vtt,
    )
    return payload, "fallback"


def transcribe_audio(
    audio_path: Path,
    source_video: Path,
    language: str,
    transcript_json: Path,
    transcript_txt: Path,
    transcript_srt: Path,
    transcript_vtt: Path,
    force: bool,
    model_name: str = WHISPER_MODEL,
) -> dict[str, object]:
    if transcript_json.exists() and not force:
        payload = read_json(transcript_json)
        if can_reuse_transcript(payload, source_video, language, model_name):
            if not transcript_txt.exists() or not transcript_srt.exists() or not transcript_vtt.exists():
                write_transcript_artifacts(
                    payload,
                    transcript_json,
                    transcript_txt,
                    transcript_srt,
                    transcript_vtt,
                )
            return payload

    import mlx_whisper

    requested_language = None if language == "auto" else language
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_name,
        language=requested_language,
        verbose=False,
    )

    raw_segments: list[tuple[float, float, str]] = []
    for segment in result["segments"]:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        raw_segments.append((float(segment["start"]), float(segment["end"]), text))

    payload = build_transcript_payload(
        str(source_video),
        str(result.get("language") or language),
        language,
        model_name,
        raw_segments,
    )
    write_transcript_artifacts(
        payload,
        transcript_json,
        transcript_txt,
        transcript_srt,
        transcript_vtt,
    )
    return payload

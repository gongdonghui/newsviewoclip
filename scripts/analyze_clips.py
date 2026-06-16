import json
import logging
import os
import re

from scripts.common import load_runtime_env

REQUIRED_CLIP_FIELDS = {
    "clip_id",
    "title",
    "start",
    "end",
    "summary",
    "keywords",
    "reason",
    "confidence",
}

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_MAX_ATTEMPTS = 3


def build_analysis_messages(
    probe_metadata: dict[str, object],
    transcript_segments: list[dict[str, object]],
) -> list[dict[str, str]]:
    system_message = (
        "You are segmenting an English news video into 15-45 second clips. "
        "Prefer broadcaster-led clips centered on the anchor. "
        "Allow only brief reporter or soundbite continuation when it sharpens the point. "
        "Prefer more clips that are tighter over longer story packages. "
        "Return strict JSON with a top-level clips array. Each clip object must include "
        "clip_id, title, start, end, summary, keywords, reason, and confidence."
    )
    user_lines = [
        "Video metadata:",
        json.dumps(probe_metadata, ensure_ascii=False),
        "Transcript segments:",
        json.dumps(transcript_segments, ensure_ascii=False),
    ]
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def parse_analysis_response(content: str) -> list[dict[str, object]]:
    payload = json.loads(_extract_json_payload(content))
    clips = payload.get("clips")
    if not isinstance(clips, list):
        raise ValueError("Expected top-level clips list")

    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("Expected each clip to be an object")

        missing_fields = REQUIRED_CLIP_FIELDS.difference(clip)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing required clip fields: {missing}")

    return clips


def _extract_json_payload(content: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    stripped_content = content.strip()
    try:
        json.loads(stripped_content)
    except json.JSONDecodeError:
        start = stripped_content.find("{")
        end = stripped_content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in analysis response") from None
        return stripped_content[start : end + 1]

    return stripped_content


def request_clip_analysis(
    probe_metadata: dict[str, object],
    transcript_segments: list[dict[str, object]],
    env: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    from openai import OpenAI

    settings = dict(load_runtime_env())
    if env is not None:
        settings.update(env)
    api_key = settings.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url=settings.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL,
    )
    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_DEEPSEEK_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=settings.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL,
                messages=build_analysis_messages(probe_metadata, transcript_segments),
                temperature=0,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty analysis response")
            return parse_analysis_response(content)
        except Exception as exc:
            last_error = exc
            if attempt == DEFAULT_DEEPSEEK_MAX_ATTEMPTS:
                break
            logging.warning(
                "DeepSeek analysis attempt %s/%s failed: %s",
                attempt,
                DEFAULT_DEEPSEEK_MAX_ATTEMPTS,
                exc,
            )

    assert last_error is not None
    raise last_error

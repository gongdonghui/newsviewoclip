import sys
from types import SimpleNamespace

import pytest

from scripts.analyze_clips import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    build_analysis_messages,
    parse_analysis_response,
    request_clip_analysis,
)


def test_build_analysis_messages_uses_expected_roles_order_and_schema_prompt():
    messages = build_analysis_messages(
        {"duration_seconds": 120.0, "resolution": "1920x1080"},
        [{"id": 1, "start": 0.0, "end": 10.0, "text": "Opening headline."}],
    )

    assert messages[0]["role"] == "system"
    assert "Return strict JSON with a top-level clips array." in messages[0]["content"]
    assert (
        "clip_id, title, start, end, summary, keywords, reason, and confidence."
        in messages[0]["content"]
    )
    assert messages[1] == {
        "role": "user",
        "content": (
            "Video metadata:\n"
            '{"duration_seconds": 120.0, "resolution": "1920x1080"}\n'
            "Transcript segments:\n"
            '[{"id": 1, "start": 0.0, "end": 10.0, "text": "Opening headline."}]'
        ),
    }


def test_build_analysis_messages_uses_broadcaster_focused_prompt_guidance():
    messages = build_analysis_messages(
        {"duration_seconds": 120.0, "resolution": "1920x1080"},
        [{"id": 1, "start": 0.0, "end": 10.0, "text": "Opening headline."}],
    )

    system_message = messages[0]["content"]

    assert "15-45" in system_message
    assert "broadcaster" in system_message
    assert "anchor" in system_message
    assert "brief reporter or soundbite continuation" in system_message
    assert "more clips that are tighter" in system_message


def test_parse_analysis_response_accepts_fenced_wrapped_json():
    content = """
Here is the clip plan:

```json
{"clips":[{"clip_id":"clip_001","title":"Opening headline","start":0.0,"end":60.0,"summary":"Lead story.","keywords":["headline"],"reason":"Complete segment.","confidence":0.9}]}
```
"""

    clips = parse_analysis_response(content)

    assert clips == [
        {
            "clip_id": "clip_001",
            "title": "Opening headline",
            "start": 0.0,
            "end": 60.0,
            "summary": "Lead story.",
            "keywords": ["headline"],
            "reason": "Complete segment.",
            "confidence": 0.9,
        }
    ]


def test_parse_analysis_response_rejects_missing_required_fields():
    content = '{"clips":[{"clip_id":"clip_001","title":"Opening headline"}]}'

    with pytest.raises(ValueError, match="Missing required clip fields"):
        parse_analysis_response(content)


def test_request_clip_analysis_uses_mocked_openai_client(monkeypatch):
    calls: dict[str, object] = {}

    class FakeCompletions:
        def create(self, *, model, messages, temperature):
            calls["model"] = model
            calls["messages"] = messages
            calls["temperature"] = temperature
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"clips":[{"clip_id":"clip_001","title":"Lead","start":0.0,"end":60.0,"summary":"Lead story.","keywords":[],"reason":"Complete.","confidence":0.9}]}'
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            calls["api_key"] = api_key
            calls["base_url"] = base_url
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    clips = request_clip_analysis(
        {"duration_seconds": 120.0, "resolution": "1920x1080"},
        [{"id": 1, "start": 0.0, "end": 10.0, "text": "Opening headline."}],
        env={"DEEPSEEK_API_KEY": "test-key"},
    )

    assert calls["api_key"] == "test-key"
    assert calls["base_url"] == DEFAULT_DEEPSEEK_BASE_URL
    assert calls["model"] == DEFAULT_DEEPSEEK_MODEL
    assert calls["temperature"] == 0
    assert clips[0]["clip_id"] == "clip_001"


def test_request_clip_analysis_retries_after_transient_failure(monkeypatch):
    calls = {"attempts": 0}

    class FakeCompletions:
        def create(self, *, model, messages, temperature):
            calls["attempts"] += 1
            if calls["attempts"] < 3:
                raise RuntimeError("temporary deepseek failure")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"clips":[{"clip_id":"clip_001","title":"Lead","start":0.0,"end":30.0,"summary":"Lead story.","keywords":[],"reason":"Complete.","confidence":0.9}]}'
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    clips = request_clip_analysis(
        {"duration_seconds": 120.0, "resolution": "1920x1080"},
        [{"id": 1, "start": 0.0, "end": 10.0, "text": "Opening headline."}],
        env={"DEEPSEEK_API_KEY": "test-key"},
    )

    assert calls["attempts"] == 3
    assert clips[0]["clip_id"] == "clip_001"


def test_request_clip_analysis_raises_after_retry_budget_exhausted(monkeypatch):
    calls = {"attempts": 0}

    class FakeCompletions:
        def create(self, *, model, messages, temperature):
            calls["attempts"] += 1
            raise RuntimeError("temporary deepseek failure")

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    with pytest.raises(RuntimeError, match="temporary deepseek failure"):
        request_clip_analysis(
            {"duration_seconds": 120.0, "resolution": "1920x1080"},
            [{"id": 1, "start": 0.0, "end": 10.0, "text": "Opening headline."}],
            env={"DEEPSEEK_API_KEY": "test-key"},
        )

    assert calls["attempts"] == 3

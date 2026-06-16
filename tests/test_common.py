from pathlib import Path

from scripts.common import (
    format_seconds,
    load_env_file,
    seconds_to_timestamp,
    timestamp_to_seconds,
)


def test_timestamp_round_trip_with_milliseconds():
    value = 72.345
    timestamp = seconds_to_timestamp(value)
    assert timestamp == "00:01:12.345"
    assert timestamp_to_seconds(timestamp) == 72.345


def test_format_seconds_uses_single_decimal_for_summary():
    assert format_seconds(53.54) == "53.5"


def test_load_env_file_reads_simple_key_values(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=test-key\nDEEPSEEK_MODEL=deepseek-chat\n", encoding="utf-8")

    values = load_env_file(env_file)

    assert values["DEEPSEEK_API_KEY"] == "test-key"
    assert values["DEEPSEEK_MODEL"] == "deepseek-chat"

from __future__ import annotations

from scripts.common import seconds_to_timestamp, timestamp_to_seconds

MEANINGFUL_GAP_SECONDS = 5.0
PAUSE_LOOKBACK_SECONDS = 1.2
PAUSE_LOOKAHEAD_SECONDS = 0.8


def clamp_clip_end(
    start: float,
    end: float,
    max_clip_seconds: float | None,
) -> float:
    if max_clip_seconds is None:
        return end
    return min(end, start + float(max_clip_seconds))


def find_pause_boundary(
    start: float,
    end: float,
    pauses: list[dict[str, float]] | None,
    min_clip_seconds: float,
    max_clip_seconds: float | None,
) -> float | None:
    if not pauses:
        return None

    capped_end = clamp_clip_end(start, end, max_clip_seconds)
    earliest_end = start + float(min_clip_seconds)
    before_candidates: list[float] = []
    after_candidates: list[float] = []

    for pause in pauses:
        pause_start = float(pause["start"])
        if pause_start < earliest_end:
            continue
        if capped_end - PAUSE_LOOKBACK_SECONDS <= pause_start <= capped_end:
            before_candidates.append(pause_start)
        elif capped_end < pause_start <= capped_end + PAUSE_LOOKAHEAD_SECONDS:
            after_candidates.append(pause_start)

    if before_candidates:
        return max(before_candidates)
    if after_candidates:
        return min(after_candidates)
    return None


def snap_clip_end(
    start: float,
    end: float,
    pauses: list[dict[str, float]] | None,
    min_clip_seconds: float,
    max_clip_seconds: float | None,
) -> float:
    capped_end = clamp_clip_end(start, end, max_clip_seconds)
    pause_boundary = find_pause_boundary(
        start,
        capped_end,
        pauses,
        min_clip_seconds,
        max_clip_seconds,
    )
    if pause_boundary is None:
        return capped_end
    if pause_boundary - start < float(min_clip_seconds):
        return capped_end
    return pause_boundary


def can_flush_at_end(
    start: float,
    end: float,
    pauses: list[dict[str, float]] | None,
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> bool:
    if not pauses:
        return True
    return (
        find_pause_boundary(
            start,
            end,
            pauses,
            min_clip_seconds,
            max_clip_seconds,
        )
        is not None
    )


def normalize_clips(
    raw_clips: list[dict[str, object]],
    duration_seconds: float,
    padding_before: float,
    padding_after: float,
    output_file_prefix: str = "output/clips",
    pauses: list[dict[str, float]] | None = None,
    min_clip_seconds: float = 0.0,
    max_clip_seconds: float | None = None,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    normalized_output_prefix = output_file_prefix.rstrip("/")
    for index, clip in enumerate(raw_clips, start=1):
        clip_id = str(clip.get("clip_id", f"clip_{index:03d}"))
        raw_start = coerce_seconds(clip["start"])
        raw_end = coerce_seconds(clip["end"])
        snapped_end = snap_clip_end(
            raw_start,
            raw_end,
            pauses,
            min_clip_seconds,
            max_clip_seconds,
        )
        start = max(0.0, raw_start - padding_before)
        end = min(duration_seconds, snapped_end + padding_after)
        if end <= start:
            continue
        duration = round(end - start, 3)
        normalized.append(
            {
                "clip_id": clip_id,
                "title": str(clip["title"]),
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
                "duration_seconds": duration,
                "summary": str(clip["summary"]),
                "keywords": list(clip.get("keywords", [])),
                "reason": str(clip["reason"]),
                "confidence": float(clip.get("confidence", 0.0)),
                "output_file": f"{normalized_output_prefix}/{clip_id}.mp4",
            }
        )
    return normalized


def coerce_seconds(value: object) -> float:
    if isinstance(value, str):
        stripped = value.strip()
        if ":" in stripped:
            return timestamp_to_seconds(stripped.replace(",", "."))
        return float(stripped)
    return float(value)


def build_fallback_clips(
    transcript: dict[str, object],
    min_clip_seconds: int,
    target_clip_seconds: int,
    max_clip_seconds: int,
    pauses: list[dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    clips: list[dict[str, object]] = []
    current: list[dict[str, object]] = []
    short_target_floor = max(float(min_clip_seconds), float(target_clip_seconds) * 0.8)

    def append_current_clip() -> None:
        if not current:
            return
        clip_start = float(current[0]["start"])
        clip_end = snap_clip_end(
            clip_start,
            float(current[-1]["end"]),
            pauses,
            min_clip_seconds,
            max_clip_seconds,
        )
        clips.append(
            {
                "title": str(current[0]["text"])[:60],
                "start": clip_start,
                "end": clip_end,
                "summary": " ".join(str(item["text"]) for item in current),
                "keywords": [],
                "reason": "Fallback semantic grouping.",
                "confidence": 0.0,
            }
        )

    for segment in transcript["segments"]:
        if current:
            current_duration = float(current[-1]["end"]) - float(current[0]["start"])
            gap = float(segment["start"]) - float(current[-1]["end"])
            if gap >= MEANINGFUL_GAP_SECONDS and current_duration >= min_clip_seconds:
                append_current_clip()
                current = []
            elif (
                current_duration >= short_target_floor
                and float(segment["end"]) - float(current[0]["start"]) > target_clip_seconds
                and can_flush_at_end(
                    float(current[0]["start"]),
                    float(current[-1]["end"]),
                    pauses,
                    min_clip_seconds,
                    max_clip_seconds,
                )
            ):
                append_current_clip()
                current = []
        current.append(segment)
        start = float(current[0]["start"])
        end = float(current[-1]["end"])
        duration = end - start
        if duration >= max_clip_seconds:
            append_current_clip()
            current = []
        elif duration >= target_clip_seconds and can_flush_at_end(
            start,
            end,
            pauses,
            min_clip_seconds,
            max_clip_seconds,
        ):
            append_current_clip()
            current = []
    if current:
        append_current_clip()
    if len(clips) >= 2:
        tail = clips[-1]
        tail_duration = float(tail["end"]) - float(tail["start"])
        if tail_duration < min_clip_seconds:
            previous = clips[-2]
            merged_duration = float(tail["end"]) - float(previous["start"])
            if merged_duration <= max_clip_seconds:
                previous["end"] = tail["end"]
                previous["summary"] = f'{previous["summary"]} {tail["summary"]}'
                clips.pop()
    return clips

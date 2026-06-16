from scripts.build_clip_plan import build_fallback_clips, normalize_clips


def test_normalize_clips_applies_padding_and_duration():
    clips = normalize_clips(
        [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 10.0,
                "end": 40.0,
                "summary": "Lead.",
                "keywords": ["lead"],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.5,
        padding_after=1.0,
    )

    assert clips[0]["start"] == "00:00:09.500"
    assert clips[0]["end"] == "00:00:41.000"
    assert clips[0]["duration_seconds"] == 31.5


def test_normalize_clips_keeps_output_file_in_sync_with_clip_id():
    clips = normalize_clips(
        [
            {
                "clip_id": "lead_story",
                "title": "Lead",
                "start": 10.0,
                "end": 40.0,
                "summary": "Lead.",
                "keywords": ["lead"],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.5,
        padding_after=1.0,
    )

    assert clips[0]["clip_id"] == "lead_story"
    assert clips[0]["output_file"] == "output/clips/lead_story.mp4"


def test_normalize_clips_supports_custom_output_file_prefix():
    clips = normalize_clips(
        [
            {
                "clip_id": "lead_story",
                "title": "Lead",
                "start": 10.0,
                "end": 40.0,
                "summary": "Lead.",
                "keywords": ["lead"],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.5,
        padding_after=1.0,
        output_file_prefix="output/example/clips",
    )

    assert clips[0]["output_file"] == "output/example/clips/lead_story.mp4"


def test_normalize_clips_snaps_end_to_nearby_pause():
    clips = normalize_clips(
        [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 10.0,
                "end": 40.5,
                "summary": "Lead.",
                "keywords": ["lead"],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.0,
        padding_after=0.0,
        pauses=[{"start": 39.8, "end": 40.4, "duration": 0.6}],
        min_clip_seconds=15,
        max_clip_seconds=45,
    )

    assert clips[0]["end"] == "00:00:39.800"
    assert clips[0]["duration_seconds"] == 29.8


def test_normalize_clips_keeps_original_end_without_pause():
    clips = normalize_clips(
        [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 10.0,
                "end": 40.5,
                "summary": "Lead.",
                "keywords": ["lead"],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.0,
        padding_after=0.0,
        pauses=[],
        min_clip_seconds=15,
        max_clip_seconds=45,
    )

    assert clips[0]["end"] == "00:00:40.500"
    assert clips[0]["duration_seconds"] == 30.5


def test_normalize_clips_does_not_snap_below_minimum_duration():
    clips = normalize_clips(
        [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 10.0,
                "end": 25.0,
                "summary": "Lead.",
                "keywords": ["lead"],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.0,
        padding_after=0.0,
        pauses=[{"start": 23.0, "end": 23.6, "duration": 0.6}],
        min_clip_seconds=15,
        max_clip_seconds=45,
    )

    assert clips[0]["end"] == "00:00:25.000"
    assert clips[0]["duration_seconds"] == 15.0


def test_normalize_clips_respects_hard_maximum_without_pause():
    clips = normalize_clips(
        [
            {
                "clip_id": "clip_001",
                "title": "Lead",
                "start": 0.0,
                "end": 50.0,
                "summary": "Lead.",
                "keywords": ["lead"],
                "reason": "Complete.",
                "confidence": 0.9,
            }
        ],
        duration_seconds=120.0,
        padding_before=0.0,
        padding_after=0.0,
        pauses=[],
        min_clip_seconds=15,
        max_clip_seconds=45,
    )

    assert clips[0]["end"] == "00:00:45.000"
    assert clips[0]["duration_seconds"] == 45.0


def test_build_fallback_clips_splits_on_meaningful_gap_after_minimum_duration():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 20.0, "text": "Opening headline."},
            {"id": 2, "start": 20.5, "end": 45.0, "text": "Reporter adds details."},
            {"id": 3, "start": 60.0, "end": 90.0, "text": "Second story begins."},
        ]
    }

    clips = build_fallback_clips(
        transcript,
        min_clip_seconds=20,
        target_clip_seconds=60,
        max_clip_seconds=180,
    )

    assert len(clips) == 2
    assert clips[0]["start"] == 0.0
    assert clips[0]["end"] == 45.0
    assert clips[1]["start"] == 60.0


def test_build_fallback_clips_merges_short_tail_when_within_maximum_duration():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 35.0, "text": "Opening headline."},
            {"id": 2, "start": 35.5, "end": 70.0, "text": "Reporter adds details."},
            {"id": 3, "start": 85.0, "end": 95.0, "text": "Brief wrap-up."},
        ]
    }

    clips = build_fallback_clips(
        transcript,
        min_clip_seconds=20,
        target_clip_seconds=60,
        max_clip_seconds=180,
    )

    assert len(clips) == 1
    assert clips[0]["start"] == 0.0
    assert clips[0]["end"] == 95.0


def test_build_fallback_clips_flushes_short_payoff_before_background_continuation():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 12.0, "text": "Anchor sets up the rescue timeline."},
            {"id": 2, "start": 12.2, "end": 28.0, "text": "Anchor confirms the child is now safe."},
            {"id": 3, "start": 28.2, "end": 41.0, "text": "Reporter recaps background from the scene."},
            {"id": 4, "start": 41.2, "end": 54.0, "text": "Reporter adds reaction from witnesses."},
        ]
    }

    clips = build_fallback_clips(
        transcript,
        min_clip_seconds=15,
        target_clip_seconds=30,
        max_clip_seconds=45,
    )

    assert len(clips) == 2
    assert clips[0]["start"] == 0.0
    assert clips[0]["end"] == 28.0
    assert clips[1]["start"] == 28.2
    assert clips[1]["end"] == 54.0


def test_build_fallback_clips_waits_for_nearby_pause_before_flushing():
    transcript = {
        "segments": [
            {"id": 1, "start": 0.0, "end": 16.0, "text": "Anchor introduces the breaking story."},
            {"id": 2, "start": 16.1, "end": 31.0, "text": "Anchor adds the key confirmed detail."},
            {"id": 3, "start": 31.1, "end": 43.0, "text": "Anchor closes with the immediate takeaway."},
        ]
    }

    clips = build_fallback_clips(
        transcript,
        min_clip_seconds=15,
        target_clip_seconds=30,
        max_clip_seconds=45,
        pauses=[{"start": 42.6, "end": 43.2, "duration": 0.6}],
    )

    assert len(clips) == 1
    assert clips[0]["start"] == 0.0
    assert clips[0]["end"] == 42.6

"""
tests/test_config.py
====================
Unit tests for config.py — SlotConfig defaults and homography round-trip.
"""

import json
import os
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import SlotConfig, Source, save_slots, load_slots


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_source(slots_file: str) -> Source:
    """Create a minimal Source pointing at an absolute path."""
    return Source(
        id="test",
        name="Test Source",
        uri="video/CarPark.mp4",
        slots_file=slots_file,
        entry_point=(0, 0),
    )


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestSlotConfigDefaults:
    def test_homography_matrix_default_is_none(self):
        cfg = SlotConfig()
        assert cfg.homography_matrix is None

    def test_warped_size_default_is_none(self):
        cfg = SlotConfig()
        assert cfg.warped_size is None

    def test_other_defaults_unchanged(self):
        cfg = SlotConfig()
        assert cfg.rect_w == 100
        assert cfg.rect_h == 33
        assert cfg.threshold == 30
        assert cfg.slots == []


class TestSaveLoadRoundTrip:
    def test_roundtrip_with_none_homography(self, tmp_path):
        """Backward compatibility: saving with None homography should load back as None."""
        slots_file = str(tmp_path / "slots.json")
        source = _make_source(slots_file)
        cfg = SlotConfig(
            rect_w=80,
            rect_h=40,
            threshold=25,
            slots=[(10, 20), (30, 40)],
            homography_matrix=None,
            warped_size=None,
        )
        save_slots(source, cfg)
        loaded = load_slots(source)

        assert loaded.rect_w == 80
        assert loaded.rect_h == 40
        assert loaded.threshold == 25
        assert loaded.slots == [(10, 20), (30, 40)]
        assert loaded.homography_matrix is None
        assert loaded.warped_size is None

    def test_roundtrip_with_homography_matrix(self, tmp_path):
        """Saving with a homography matrix should load it back correctly."""
        slots_file = str(tmp_path / "slots.json")
        source = _make_source(slots_file)
        matrix = [[1.2, 0.0, -50.0], [0.0, 1.1, -20.0], [0.0, 0.0, 1.0]]
        cfg = SlotConfig(
            slots=[(5, 5)],
            homography_matrix=matrix,
            warped_size=[960, 540],
        )
        save_slots(source, cfg)
        loaded = load_slots(source)

        assert loaded.homography_matrix == matrix
        assert loaded.warped_size == [960, 540]


class TestLoadSlotsBackwardCompatibility:
    def test_missing_homography_key_returns_none(self, tmp_path):
        """A JSON file without 'homography_matrix' key should load with None."""
        slots_file = str(tmp_path / "slots_old.json")
        # Write a JSON that has no homography_matrix key (old format)
        old_data = {
            "rect_w": 100,
            "rect_h": 33,
            "threshold": 30,
            "slots": [[100, 200], [300, 400]],
        }
        with open(slots_file, "w", encoding="utf-8") as f:
            json.dump(old_data, f)

        source = _make_source(slots_file)
        loaded = load_slots(source)

        assert loaded.homography_matrix is None
        assert loaded.warped_size is None
        assert loaded.slots == [(100, 200), (300, 400)]

    def test_missing_warped_size_key_returns_none(self, tmp_path):
        """A JSON file without 'warped_size' key should load with None."""
        slots_file = str(tmp_path / "slots_partial.json")
        partial_data = {
            "rect_w": 100,
            "rect_h": 33,
            "threshold": 30,
            "slots": [],
            "homography_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            # no warped_size key
        }
        with open(slots_file, "w", encoding="utf-8") as f:
            json.dump(partial_data, f)

        source = _make_source(slots_file)
        loaded = load_slots(source)

        assert loaded.homography_matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert loaded.warped_size is None

    def test_missing_file_returns_empty_slotconfig(self, tmp_path):
        """load_slots on a non-existent file returns a default SlotConfig."""
        slots_file = str(tmp_path / "nonexistent.json")
        source = _make_source(slots_file)
        loaded = load_slots(source)

        assert loaded.homography_matrix is None
        assert loaded.warped_size is None
        assert loaded.slots == []

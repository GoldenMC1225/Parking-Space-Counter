"""
tests/test_evaluate.py
======================
Unit tests for evaluate.py — load_ground_truth.
"""

import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import load_ground_truth


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def write_csv(tmp_path, content: str) -> str:
    """Write *content* to a temp CSV file and return its path."""
    p = tmp_path / "gt.csv"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


# ─────────────────────────────────────────────
# load_ground_truth — file-level errors
# ─────────────────────────────────────────────

class TestLoadGroundTruthFileErrors:
    def test_exits_1_on_missing_file(self):
        """Req 7.7 / 8.1: missing file → sys.exit(1)."""
        with pytest.raises(SystemExit) as exc_info:
            load_ground_truth("/nonexistent/path/gt.csv", n_slots=5)
        assert exc_info.value.code == 1

    def test_exits_1_on_missing_all_columns(self, tmp_path):
        """Req 8.1: CSV with wrong header → sys.exit(1)."""
        path = write_csv(tmp_path, """\
            wrong_col_a,wrong_col_b
            1,2
        """)
        with pytest.raises(SystemExit) as exc_info:
            load_ground_truth(path, n_slots=5)
        assert exc_info.value.code == 1

    def test_exits_1_on_missing_one_column(self, tmp_path):
        """Req 8.1: CSV missing just 'occupied' → sys.exit(1)."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id
            0,0
        """)
        with pytest.raises(SystemExit) as exc_info:
            load_ground_truth(path, n_slots=5)
        assert exc_info.value.code == 1

    def test_exits_1_on_empty_file(self, tmp_path):
        """Empty file (no header) → sys.exit(1)."""
        path = write_csv(tmp_path, "")
        with pytest.raises(SystemExit) as exc_info:
            load_ground_truth(path, n_slots=5)
        assert exc_info.value.code == 1


# ─────────────────────────────────────────────
# load_ground_truth — valid data
# ─────────────────────────────────────────────

class TestLoadGroundTruthValidData:
    def test_returns_correct_mapping(self, tmp_path):
        """Happy path: valid CSV returns correct nested dict."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,0,1
            0,1,0
            1,0,0
        """)
        result = load_ground_truth(path, n_slots=3)
        assert result == {0: {0: 1, 1: 0}, 1: {0: 0}}

    def test_returns_empty_dict_for_header_only(self, tmp_path):
        """CSV with header but no data rows → empty dict."""
        path = write_csv(tmp_path, "frame_index,slot_id,occupied\n")
        result = load_ground_truth(path, n_slots=5)
        assert result == {}

    def test_extra_columns_are_ignored(self, tmp_path):
        """Extra columns beyond the required three are silently ignored."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied,extra_col
            0,0,1,foo
        """)
        result = load_ground_truth(path, n_slots=3)
        assert result == {0: {0: 1}}


# ─────────────────────────────────────────────
# load_ground_truth — row-level validation
# ─────────────────────────────────────────────

class TestLoadGroundTruthRowValidation:
    def test_skips_row_with_slot_id_out_of_range_high(self, tmp_path):
        """Req 8.2: slot_id >= n_slots → row skipped, rest processed."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,5,1
            0,0,1
        """)
        result = load_ground_truth(path, n_slots=5)  # valid range [0,4]
        assert 5 not in result.get(0, {})
        assert result[0][0] == 1

    def test_skips_row_with_slot_id_negative(self, tmp_path):
        """Req 8.2: negative slot_id → row skipped."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,-1,1
            0,0,0
        """)
        result = load_ground_truth(path, n_slots=3)
        assert -1 not in result.get(0, {})
        assert result[0][0] == 0

    def test_skips_row_with_occupied_value_2(self, tmp_path):
        """Req 8.3: occupied=2 → row skipped, rest processed."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,0,2
            0,1,1
        """)
        result = load_ground_truth(path, n_slots=3)
        assert 0 not in result.get(0, {})
        assert result[0][1] == 1

    def test_skips_row_with_occupied_value_minus_1(self, tmp_path):
        """Req 8.3: occupied=-1 → row skipped."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,0,-1
            0,1,0
        """)
        result = load_ground_truth(path, n_slots=3)
        assert 0 not in result.get(0, {})
        assert result[0][1] == 0

    def test_multiple_invalid_rows_all_skipped(self, tmp_path):
        """Multiple bad rows are all skipped; valid rows are kept."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,99,1
            0,0,5
            1,0,1
        """)
        result = load_ground_truth(path, n_slots=5)
        assert 0 not in result  # both frame-0 rows were invalid
        assert result[1][0] == 1

    def test_boundary_slot_id_zero_is_valid(self, tmp_path):
        """slot_id=0 is the lower boundary and must be accepted."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,0,1
        """)
        result = load_ground_truth(path, n_slots=3)
        assert result[0][0] == 1

    def test_boundary_slot_id_n_slots_minus_1_is_valid(self, tmp_path):
        """slot_id = n_slots-1 is the upper boundary and must be accepted."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,2,0
        """)
        result = load_ground_truth(path, n_slots=3)
        assert result[0][2] == 0

    def test_occupied_0_and_1_both_valid(self, tmp_path):
        """Both occupied=0 and occupied=1 are valid values."""
        path = write_csv(tmp_path, """\
            frame_index,slot_id,occupied
            0,0,0
            0,1,1
        """)
        result = load_ground_truth(path, n_slots=3)
        assert result[0][0] == 0
        assert result[0][1] == 1


from evaluate import compute_metrics


# ─────────────────────────────────────────────
# compute_metrics — unit tests
# ─────────────────────────────────────────────

class TestComputeMetrics:
    # ── Division-by-zero / empty cases ──────────────────────────────────

    def test_returns_zeros_when_ground_truth_empty(self):
        """Empty ground-truth → (0.0, 0.0, 0.0); nothing to evaluate."""
        assert compute_metrics([True, False], {}) == (0.0, 0.0, 0.0)

    def test_returns_zeros_when_no_positive_predictions(self):
        """All predicted free, all GT occupied → precision=0, recall=0, f1=0."""
        predicted = [False, False]
        gt = {0: 1, 1: 1}
        precision, recall, f1 = compute_metrics(predicted, gt)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_returns_zeros_when_no_gt_occupied_and_all_predicted_free(self):
        """All GT free, all predicted free → TP=FP=FN=0 → precision=0, recall=0."""
        predicted = [False, False]
        gt = {0: 0, 1: 0}
        precision, recall, f1 = compute_metrics(predicted, gt)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    # ── Perfect predictions ──────────────────────────────────────────────

    def test_perfect_all_occupied(self):
        """All slots occupied, all predicted occupied → precision=recall=f1=1.0."""
        predicted = [True, True, True]
        gt = {0: 1, 1: 1, 2: 1}
        precision, recall, f1 = compute_metrics(predicted, gt)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_perfect_mixed(self):
        """Mixed GT, predictions match exactly → precision=recall=f1=1.0."""
        predicted = [True, False, True, False]
        gt = {0: 1, 1: 0, 2: 1, 3: 0}
        precision, recall, f1 = compute_metrics(predicted, gt)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    # ── Known TP/FP/FN values ────────────────────────────────────────────

    def test_known_tp_fp_fn(self):
        """TP=1, FP=1, FN=1, TN=1 → precision=0.5, recall=0.5, f1=0.5."""
        # slot 0: predicted=T, gt=1 → TP
        # slot 1: predicted=T, gt=0 → FP
        # slot 2: predicted=F, gt=1 → FN
        # slot 3: predicted=F, gt=0 → TN
        predicted = [True, True, False, False]
        gt = {0: 1, 1: 0, 2: 1, 3: 0}
        precision, recall, f1 = compute_metrics(predicted, gt)
        assert abs(precision - 0.5) < 1e-9
        assert abs(recall - 0.5) < 1e-9
        expected_f1 = 2 * 0.5 * 0.5 / (0.5 + 0.5)
        assert abs(f1 - expected_f1) < 1e-9

    def test_precision_zero_recall_one(self):
        """All GT occupied, all predicted occupied but one extra FP → recall=1."""
        # slot 0: T, gt=1 → TP
        # slot 1: T, gt=0 → FP
        predicted = [True, True]
        gt = {0: 1, 1: 0}
        precision, recall, f1 = compute_metrics(predicted, gt)
        assert precision == 0.5
        assert recall == 1.0
        assert abs(f1 - 2 * 0.5 * 1.0 / 1.5) < 1e-9

    # ── Req 8.4: slots absent from GT are excluded ───────────────────────

    def test_slots_absent_from_gt_are_excluded(self):
        """Slots not in ground_truth_frame must not affect any metric."""
        # predicted has 5 slots; GT only covers slots 0 and 1
        predicted = [True, False, True, True, False]
        gt = {0: 1, 1: 0}  # slots 2,3,4 absent → excluded
        # slot 0: T, gt=1 → TP; slot 1: F, gt=0 → TN
        # TP=1, FP=0, FN=0 → precision=1.0, recall=1.0
        precision, recall, f1 = compute_metrics(predicted, gt)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    # ── Out-of-range slot_id in GT ───────────────────────────────────────

    def test_slot_id_out_of_range_is_skipped(self):
        """GT slot_id >= len(predicted) must be silently skipped."""
        predicted = [True]  # only slot 0 exists
        gt = {0: 1, 5: 1}   # slot 5 is out of range
        precision, recall, f1 = compute_metrics(predicted, gt)
        # Only slot 0 counts: TP=1 → precision=recall=f1=1.0
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    # ── Return type ──────────────────────────────────────────────────────

    def test_return_type_is_tuple_of_floats(self):
        """Return value must be a 3-tuple of floats."""
        result = compute_metrics([True], {0: 1})
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

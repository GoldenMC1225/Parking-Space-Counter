"""
evaluate.py
===========
Evaluation framework for comparing the Classical and YOLO detection pipelines
against a ground-truth CSV annotation file.

Ground-truth CSV format:
    frame_index,slot_id,occupied
    0,0,1
    0,1,0
    1,0,1
    ...

Usage:
    python evaluate.py --source ID --ground-truth PATH [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

REQUIRED_COLUMNS = {"frame_index", "slot_id", "occupied"}

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Ground-truth loader
# ─────────────────────────────────────────────

def load_ground_truth(path: str, n_slots: int) -> Dict[int, Dict[int, int]]:
    """Load and validate a ground-truth CSV file.

    Parameters
    ----------
    path:
        Filesystem path to the ground-truth CSV file.
    n_slots:
        Total number of parking slots for the source.  Valid ``slot_id``
        values are in the range ``[0, n_slots - 1]``.

    Returns
    -------
    Dict[int, Dict[int, int]]
        Nested mapping ``{frame_index: {slot_id: occupied}}`` containing
        only the rows that passed all validation checks.

    Exits
    -----
    Calls ``sys.exit(1)`` if the file is missing, unreadable, or lacks any
    of the required columns (``frame_index``, ``slot_id``, ``occupied``).
    """
    # ── 1. Open the file ────────────────────────────────────────────────
    try:
        file_handle = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        logger.error("Ground-truth file not found: %s", path)
        sys.exit(1)
    except OSError as exc:
        logger.error("Cannot read ground-truth file '%s': %s", path, exc)
        sys.exit(1)

    result: Dict[int, Dict[int, int]] = defaultdict(dict)

    with file_handle:
        reader = csv.DictReader(file_handle)

        # ── 2. Validate header ───────────────────────────────────────────
        # DictReader populates fieldnames on first access (after __next__ or
        # by reading the header row explicitly).
        fieldnames = reader.fieldnames
        if fieldnames is None:
            # Empty file — treat as missing all columns.
            logger.error(
                "Ground-truth file '%s' is empty or has no header row. "
                "Missing required columns: %s",
                path,
                ", ".join(sorted(REQUIRED_COLUMNS)),
            )
            sys.exit(1)

        present_columns = set(fieldnames)
        missing_columns = REQUIRED_COLUMNS - present_columns
        if missing_columns:
            logger.error(
                "Ground-truth file '%s' is missing required column(s): %s",
                path,
                ", ".join(sorted(missing_columns)),
            )
            sys.exit(1)

        # ── 3. Process data rows ─────────────────────────────────────────
        for row_number, row in enumerate(reader, start=2):  # row 1 = header
            raw_frame = row["frame_index"].strip()
            raw_slot = row["slot_id"].strip()
            raw_occupied = row["occupied"].strip()

            # Parse frame_index
            try:
                frame_index = int(raw_frame)
            except ValueError:
                logger.warning(
                    "Row %d: cannot parse frame_index %r as int — skipping.",
                    row_number,
                    raw_frame,
                )
                continue

            # Parse slot_id
            try:
                slot_id = int(raw_slot)
            except ValueError:
                logger.warning(
                    "Row %d: cannot parse slot_id %r as int — skipping.",
                    row_number,
                    raw_slot,
                )
                continue

            # Validate slot_id range  (Req 8.2)
            if slot_id < 0 or slot_id > n_slots - 1:
                logger.warning(
                    "Row %d: slot_id %d is out of range [0, %d] — skipping.",
                    row_number,
                    slot_id,
                    n_slots - 1,
                )
                continue

            # Parse occupied
            try:
                occupied = int(raw_occupied)
            except ValueError:
                logger.error(
                    "Row %d: cannot parse occupied %r as int — skipping.",
                    row_number,
                    raw_occupied,
                )
                continue

            # Validate occupied value  (Req 8.3)
            if occupied not in (0, 1):
                logger.error(
                    "Row %d: occupied value %d is not 0 or 1 — skipping.",
                    row_number,
                    occupied,
                )
                continue

            # All checks passed — store the entry
            result[frame_index][slot_id] = occupied

    return dict(result)


# ─────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────

def compute_metrics(
    predicted: List[bool],
    ground_truth_frame: Dict[int, int],
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 for a single frame.

    Parameters
    ----------
    predicted:
        Per-slot occupancy predictions indexed by slot id.
        ``True`` means the slot is predicted as **occupied**.
    ground_truth_frame:
        Ground-truth occupancy for this frame: ``{slot_id: occupied}``
        where ``occupied`` is ``1`` (occupied) or ``0`` (free).
        Only slots present in this dict are included in the calculation;
        all other slots are excluded (Req 8.4).

    Returns
    -------
    Tuple[float, float, float]
        ``(precision, recall, f1)`` — each in ``[0.0, 1.0]``.
        Returns ``0.0`` for any metric whose denominator is zero.
    """
    tp = fp = fn = tn = 0

    for slot_id, gt_occupied in ground_truth_frame.items():
        # Skip slots whose index is out of range for the predicted list
        if slot_id >= len(predicted):
            continue

        predicted_occupied: bool = predicted[slot_id]

        if predicted_occupied and gt_occupied == 1:
            tp += 1
        elif predicted_occupied and gt_occupied == 0:
            fp += 1
        elif not predicted_occupied and gt_occupied == 1:
            fn += 1
        else:  # not predicted_occupied and gt_occupied == 0
            tn += 1

    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1: float        = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (precision, recall, f1)


# ─────────────────────────────────────────────
# Pipeline evaluation
# ─────────────────────────────────────────────

def run_pipeline_eval(
    source,
    pipeline: str,
    ground_truth: Dict[int, Dict[int, int]],
) -> List[Dict]:
    """Process each frame, compare pipeline output against ground-truth.

    Parameters
    ----------
    source:
        A :class:`config.Source` instance describing the video source.
    pipeline:
        Either ``"classical"`` or ``"yolo"``.
    ground_truth:
        Nested mapping ``{frame_index: {slot_id: occupied}}`` as returned
        by :func:`load_ground_truth`.

    Returns
    -------
    List[Dict]
        Per-frame result dicts with keys ``frame``, ``pipeline``,
        ``precision``, ``recall``, ``f1_score``, and ``fps`` (pipeline
        average, same value repeated for every row).

    Notes
    -----
    * Frames whose ``frame_index`` is not present in *ground_truth* are
      skipped and excluded from the FPS calculation (Req 7.5).
    * Classical pipeline: ``check_slots`` returns ``True = free``; we
      invert to ``True = occupied`` before calling :func:`compute_metrics`.
    * YOLO pipeline: ``assign_slots`` returns ``True = occupied`` directly.
    """
    from config import load_slots

    # ── Lazy imports so the module can be imported without ultralytics ──
    if pipeline == "classical":
        from detect_improved import convert_grayscale, check_slots
    elif pipeline == "yolo":
        from detect_yolo import load_model, warp_frame, filter_detections, assign_slots
    else:
        raise ValueError(f"Unknown pipeline: {pipeline!r}. Use 'classical' or 'yolo'.")

    cfg = load_slots(source)

    # ── YOLO-specific setup ──────────────────────────────────────────────
    if pipeline == "yolo":
        H: "np.ndarray | None" = None
        warped_size: "tuple[int, int] | None" = None
        if cfg.homography_matrix is not None:
            H = np.array(cfg.homography_matrix, dtype=np.float64)
            if cfg.warped_size is not None:
                warped_size = (int(cfg.warped_size[0]), int(cfg.warped_size[1]))
        model = load_model("yolov8n.pt")

    # ── Open video capture ───────────────────────────────────────────────
    cap = cv2.VideoCapture(source.uri)
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(source.uri))
        except (ValueError, TypeError):
            pass
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source.uri)
        return []

    results: List[Dict] = []
    frame_index: int = 0
    total_wall_time: float = 0.0
    evaluated_frames: int = 0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break  # end of video file

        # ── Skip frames with no ground-truth rows (Req 7.5) ─────────────
        if frame_index not in ground_truth:
            frame_index += 1
            continue

        gt_frame = ground_truth[frame_index]

        # ── Process frame and measure wall-clock time (Req 7.3) ─────────
        t_start = time.perf_counter()

        if pipeline == "classical":
            gray = convert_grayscale(raw_frame)
            statuses_raw, _ = check_slots(gray, cfg)
            # classical: True = free → invert to True = occupied
            statuses: List[bool] = [not s for s in statuses_raw]

        else:  # yolo
            if H is not None and warped_size is not None:
                frame = warp_frame(raw_frame, H, warped_size)
            else:
                frame = warp_frame(raw_frame, None, (0, 0))
            raw_dets = model(frame, verbose=False)[0]
            filtered = filter_detections(raw_dets, conf_thresh=0.25)
            statuses = assign_slots(filtered, cfg)

        t_end = time.perf_counter()
        total_wall_time += t_end - t_start
        evaluated_frames += 1

        precision, recall, f1 = compute_metrics(statuses, gt_frame)
        results.append({
            "frame": frame_index,
            "pipeline": pipeline,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            # fps placeholder — filled in after the loop
        })

        frame_index += 1

    cap.release()

    # ── Compute average FPS (Req 7.3) ────────────────────────────────────
    avg_fps: float = (
        evaluated_frames / total_wall_time if total_wall_time > 0.0 else 0.0
    )

    # ── Attach fps to every row (Req 7.4) ────────────────────────────────
    for row in results:
        row["fps"] = round(avg_fps, 4)

    logger.info(
        "Pipeline '%s': evaluated %d frames, avg FPS = %.2f",
        pipeline,
        evaluated_frames,
        avg_fps,
    )
    return results


# ─────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────

def write_report(results: List[Dict], output_path: str) -> None:
    """Write evaluation results to a CSV file.

    Parameters
    ----------
    results:
        Combined list of per-frame result dicts (from one or more pipeline
        runs).  Each dict must contain the keys ``frame``, ``pipeline``,
        ``precision``, ``recall``, ``f1_score``, and ``fps``.
    output_path:
        Filesystem path for the output CSV file.  The file is created or
        overwritten.

    Notes
    -----
    Columns are written in the order specified by Req 7.4:
    ``frame``, ``pipeline``, ``precision``, ``recall``, ``f1_score``, ``fps``.
    """
    fieldnames = ["frame", "pipeline", "precision", "recall", "f1_score", "fps"]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Report written to '%s' (%d rows).", output_path, len(results))


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Classical and YOLO parking-space detection pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python evaluate.py --source carpark_main "
            "--ground-truth gt.csv --output report.csv\n"
        ),
    )
    parser.add_argument(
        "--source",
        metavar="ID",
        required=True,
        help="Source ID from sources.json",
    )
    parser.add_argument(
        "--ground-truth",
        metavar="PATH",
        required=True,
        dest="ground_truth",
        help="Path to the ground-truth CSV file",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default="evaluation_report.csv",
        help="Output CSV path (default: evaluation_report.csv)",
    )

    args = parser.parse_args()

    # ── Load source ──────────────────────────────────────────────────────
    from config import load_sources, get_source_by_id, load_slots

    sources = load_sources()
    if not sources:
        logger.error("No sources defined in sources.json. Run: python config.py --add")
        sys.exit(1)

    source = get_source_by_id(args.source)
    if source is None:
        logger.error("Source '%s' not found in sources.json.", args.source)
        sys.exit(1)

    # ── Load ground truth ────────────────────────────────────────────────
    cfg = load_slots(source)
    n_slots = len(cfg.slots)
    ground_truth = load_ground_truth(args.ground_truth, n_slots=n_slots)

    # ── Run both pipelines sequentially (Req 7.8) ────────────────────────
    all_results: List[Dict] = []

    for pipeline_name in ("classical", "yolo"):
        logger.info("Running pipeline: %s", pipeline_name)
        pipeline_results = run_pipeline_eval(source, pipeline_name, ground_truth)
        all_results.extend(pipeline_results)

    # ── Write combined report (Req 7.4) ──────────────────────────────────
    write_report(all_results, args.output)
    print(f"Evaluation complete. Report saved to: {args.output}")

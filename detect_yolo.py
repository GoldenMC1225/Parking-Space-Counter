"""
detect_yolo.py
==============
YOLO-based parking space detection pipeline.

This module provides:
  - FrameCache  : dataclass holding per-slot occupancy statuses and raw
                  bounding boxes from the most recent YOLO inference call.
  - load_model  : load a YOLO model from a path or model name, exiting with
                  a non-zero status code if the file cannot be found.
  - VEHICLE_CLASSES : set of COCO class names treated as vehicles.

Functions warp_frame, filter_detections, bottom_center, assign_slots, and
run() are implemented in subsequent tasks (6.2 – 6.6).

CLI (once fully implemented):
    python detect_yolo.py [--source ID] [--model PATH] [--skip N]
                          [--conf F] [--list]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import SlotConfig, Source, load_slots, load_sources, get_source_by_id, list_sources  # noqa: F401
from utils import find_nearest_free_slot, build_obstacle_grid, astar, draw_slots, apply_crop

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

#: COCO class names that are treated as vehicles by the detection pipeline.
VEHICLE_CLASSES: set = {"car", "truck", "bus", "motorcycle"}

# ─────────────────────────────────────────────
# FrameCache
# ─────────────────────────────────────────────


@dataclass
class FrameCache:
    """Cache holding the results of the most recent YOLO inference call.

    Attributes
    ----------
    statuses : List[bool]
        One boolean per parking slot.  ``True`` means the slot is occupied,
        ``False`` means it is free.
    boxes : List[Tuple[int, int, int, int, str, float]]
        Raw bounding boxes from the last inference run, each represented as
        ``(x1, y1, x2, y2, class_name, confidence)``.
    is_initialized : bool
        ``False`` until the first YOLO inference has been executed.  While
        ``False`` the pipeline uses the initial all-free statuses.
    """

    statuses: List[bool] = field(default_factory=list)
    boxes: List[Tuple[int, int, int, int, str, float]] = field(
        default_factory=list
    )
    # Fix 1: keep original-frame boxes separately for drawing on raw_frame
    boxes_original: List[Tuple[int, int, int, int, str, float]] = field(
        default_factory=list
    )
    is_initialized: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def make(cls, n_slots: int) -> "FrameCache":
        """Create a FrameCache initialised for *n_slots* parking slots.

        All slots are marked as free (``False``) and the bounding-box list
        is empty, satisfying Requirement 4.3 (initial state before any
        inference has been run).

        Parameters
        ----------
        n_slots:
            Number of parking slots managed by the pipeline.

        Returns
        -------
        FrameCache
            A freshly initialised cache with
            ``statuses=[False] * n_slots``, ``boxes=[]``, and
            ``is_initialized=False``.
        """
        return cls(
            statuses=[False] * n_slots,
            boxes=[],
            is_initialized=False,
        )


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────


def _is_explicit_path(model_path: str) -> bool:
    """Return ``True`` when *model_path* looks like a file-system path.

    A path is considered explicit when it contains a directory separator
    (``/`` or ``\\``) or when it is an absolute path.  Bare model names
    such as ``"yolov8n.pt"`` are *not* explicit paths — ultralytics will
    auto-download them.
    """
    return (
        os.sep in model_path
        or "/" in model_path
        or "\\" in model_path
        or os.path.isabs(model_path)
    )


def load_model(model_path: str) -> YOLO:
    """Load a YOLO model and return it.

    For bare model names (e.g. ``"yolov8n.pt"``) ultralytics handles
    auto-downloading; this function simply delegates to ``YOLO(model_path)``.

    For explicit file-system paths (containing ``/``, ``\\``, or an
    absolute prefix) the file must exist on disk.  If it does not,
    an error is logged and the process exits with status code 1
    (Requirement 3.1).

    Parameters
    ----------
    model_path:
        Either a bare model name understood by ultralytics (e.g.
        ``"yolov8n.pt"``) or an explicit path to a ``.pt`` weights file.

    Returns
    -------
    YOLO
        The loaded YOLO model instance.

    Raises
    ------
    SystemExit
        With exit code 1 when *model_path* is an explicit path that does
        not exist on disk.
    """
    if _is_explicit_path(model_path) and not os.path.exists(model_path):
        _log.error(
            "Model file not found: '%s'. "
            "Provide a valid path or a bare model name such as 'yolov8n.pt'.",
            model_path,
        )
        sys.exit(1)

    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        _log.error(
            "Model file not found: '%s'.",
            model_path,
        )
        sys.exit(1)

    _log.info("Loaded YOLO model from '%s'.", model_path)
    return model


# ─────────────────────────────────────────────
# Perspective warp
# ─────────────────────────────────────────────

#: Module-level flag so the "no homography" warning is emitted only once.
_warp_warned: bool = False


def warp_frame(
    frame: np.ndarray,
    H: Optional[np.ndarray],
    size: Tuple[int, int],
) -> np.ndarray:
    """Apply a perspective warp to *frame* using homography matrix *H*.

    Parameters
    ----------
    frame:
        The raw input frame (BGR numpy array).
    H:
        A 3×3 homography matrix produced by ``cv2.getPerspectiveTransform``
        or ``cv2.findHomography``.  When ``None`` the original frame is
        returned unchanged and a warning is logged once (Requirement 3.3).
    size:
        ``(width, height)`` of the output warped frame (Requirement 3.2).

    Returns
    -------
    np.ndarray
        The warped frame, or the original frame when *H* is ``None``.
    """
    global _warp_warned

    if H is None:
        if not _warp_warned:
            _log.warning(
                "homography_matrix is None — using original frame without "
                "perspective correction (Requirement 3.3)."
            )
            _warp_warned = True
        return frame

    return cv2.warpPerspective(frame, H, size)


# ─────────────────────────────────────────────
# Detection filtering
# ─────────────────────────────────────────────


def filter_detections(
    raw_results,
    conf_thresh: float,
) -> List[Tuple[int, int, int, int, str, float]]:
    """Filter raw YOLO results to vehicle detections above a confidence threshold.

    Parameters
    ----------
    raw_results:
        The first element of ``model(frame, verbose=False)`` — a YOLO
        ``Results`` object whose ``.boxes`` attribute holds all detections.
    conf_thresh:
        Minimum confidence score (inclusive) required to keep a detection
        (Requirement 3.7).

    Returns
    -------
    List[Tuple[int, int, int, int, str, float]]
        Each entry is ``(x1, y1, x2, y2, class_name, confidence)`` for
        detections whose class is in ``VEHICLE_CLASSES`` and whose
        confidence is ≥ *conf_thresh*.
    """
    results: List[Tuple[int, int, int, int, str, float]] = []

    boxes = raw_results.boxes
    if boxes is None:
        return results

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name: str = raw_results.names[cls_id]
        conf: float = float(box.conf[0])

        if class_name not in VEHICLE_CLASSES:
            continue
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        results.append((x1, y1, x2, y2, class_name, conf))

    return results


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────


def bottom_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """Return the bottom-center point of a bounding box.

    Computes ``(int((x1 + x2) / 2), int(y2))`` as specified in
    Requirement 3.8.

    Parameters
    ----------
    x1, y1:
        Top-left corner of the bounding box.
    x2, y2:
        Bottom-right corner of the bounding box.

    Returns
    -------
    Tuple[int, int]
        ``(bx, by)`` where ``bx`` is the horizontal midpoint and ``by``
        is the bottom edge of the box.
    """
    return (int((x1 + x2) / 2), int(y2))


# ─────────────────────────────────────────────
# Fix 1: project detections from original → warped space
# ─────────────────────────────────────────────


def project_boxes_to_warped(
    boxes: List[Tuple[int, int, int, int, str, float]],
    H: np.ndarray,
) -> List[Tuple[int, int, int, int, str, float]]:
    """Project bbox center AND bottom-center through homography H.

    Returns boxes as (center_x, center_y, bottom_cx, bottom_cy, cls, conf)
    stored in the (x1,y1,x2,y2) slots so assign_slots can test both points.
    x1,y1 = projected bbox center
    x2,y2 = projected bottom-center
    """
    if not boxes:
        return []

    # Stack both center and bottom-center points for all boxes
    pts = np.array(
        [
            [int((x1 + x2) / 2), int((y1 + y2) / 2)]   # bbox center
            for x1, y1, x2, y2, *_ in boxes
        ] + [
            [int((x1 + x2) / 2), int(y2)]               # bottom-center
            for x1, y1, x2, y2, *_ in boxes
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    n = len(boxes)

    result: List[Tuple[int, int, int, int, str, float]] = []
    for i, (x1, y1, x2, y2, cls, conf) in enumerate(boxes):
        # Center stored as (x1,y1), bottom-center as (x2,y2)
        cx, cy = int(projected[i][0]),     int(projected[i][1])      # center
        bx, by = int(projected[n + i][0]), int(projected[n + i][1])  # bottom-ctr
        result.append((cx, cy, bx, by, cls, conf))
    return result


# ─────────────────────────────────────────────
# Slot assignment
# ─────────────────────────────────────────────


def assign_slots(
    boxes: List[Tuple[int, int, int, int, str, float]],
    cfg: SlotConfig,
) -> List[bool]:
    """Determine occupancy for every slot (rect and polygon).

    For warped boxes (from project_boxes_to_warped):
      x1,y1 = projected bbox center
      x2,y2 = projected bottom-center

    For non-warped boxes (no homography):
      x1,y1,x2,y2 = original bbox corners, so we compute both points inline.

    Rectangle slots: test center OR bottom-center inside the slot rect.
    Polygon slots: test center OR bottom-center inside the polygon.

    Returns a combined List[bool] — rect slots first, then poly slots.
    True = occupied.
    """
    if not boxes:
        return [False] * (len(cfg.slots) + len(cfg.poly_slots))

    # Unpack test points — works for both warped (degenerate) and real bboxes
    test_points: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []
    for x1, y1, x2, y2, *_ in boxes:
        center     = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        bot_center = (int((x1 + x2) / 2), int(y2))
        test_points.append((center, bot_center))

    statuses: List[bool] = []

    # Rectangle slots — center OR bottom-center inside rect
    for slot_x, slot_y in cfg.slots:
        w, h = cfg.rect_w, cfg.rect_h
        occupied = any(
            (slot_x <= cx < slot_x + w and slot_y <= cy < slot_y + h)
            or (slot_x <= bx < slot_x + w and slot_y <= by < slot_y + h)
            for (cx, cy), (bx, by) in test_points
        )
        statuses.append(occupied)

    # Polygon slots — center OR bottom-center inside polygon
    for poly in cfg.poly_slots:
        if len(poly) < 3:
            statuses.append(False)
            continue
        contour = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        occupied = any(
            cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0
            or cv2.pointPolygonTest(contour, (float(bx), float(by)), False) >= 0
            for (cx, cy), (bx, by) in test_points
        )
        statuses.append(occupied)

    return statuses


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

__all__ = [
    "VEHICLE_CLASSES",
    "FrameCache",
    "load_model",
    "warp_frame",
    "filter_detections",
    "bottom_center",
    "project_boxes_to_warped",
    "assign_slots",
    "run",
]


# ─────────────────────────────────────────────
# CLI argument validation helpers
# ─────────────────────────────────────────────

_DEFAULT_SKIP: int = 5
_DEFAULT_CONF: float = 0.25


def _validate_skip(val: int) -> int:
    """Validate the ``--skip`` argument.

    Returns *val* unchanged when it is in ``[1, 60]``.  Otherwise logs an
    error and returns the default value of 5 (Requirements 3.6, 4.5).

    Parameters
    ----------
    val:
        The integer value supplied via ``--skip``.

    Returns
    -------
    int
        A valid skip interval in ``[1, 60]``.
    """
    if 1 <= val <= 60:
        return val
    _log.error(
        "--skip value %d is outside the valid range [1, 60]. "
        "Using default value %d.",
        val,
        _DEFAULT_SKIP,
    )
    return _DEFAULT_SKIP


def _validate_conf(val: float) -> float:
    """Validate the ``--conf`` argument.

    Returns *val* unchanged when it is in ``(0, 1]``.  Otherwise logs an
    error and returns the default value of 0.25 (Requirement 3.6).

    Parameters
    ----------
    val:
        The float value supplied via ``--conf``.

    Returns
    -------
    float
        A valid confidence threshold in ``(0, 1]``.
    """
    if 0.0 < val <= 1.0:
        return val
    _log.error(
        "--conf value %.4f is outside the valid range (0, 1]. "
        "Using default value %.2f.",
        val,
        _DEFAULT_CONF,
    )
    return _DEFAULT_CONF


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────


def run(
    source: Source,
    model_path: str = "yolov8n.pt",
    skip: int = _DEFAULT_SKIP,
    conf: float = _DEFAULT_CONF,
) -> None:
    """Run the YOLO-based parking space detection pipeline.

    Loads the slot configuration and YOLO model, opens the video capture,
    and enters the main frame-processing loop.  On every *skip*-th frame
    YOLO inference is executed and the Frame_Cache is updated; on all other
    frames the cached results are reused.

    Parameters
    ----------
    source:
        The video/RTSP source to process.
    model_path:
        Path to the YOLO weights file, or a bare model name understood by
        ultralytics (e.g. ``"yolov8n.pt"``).
    skip:
        Frame-skip interval — inference runs when
        ``frame_index % skip == 0``.  Must be in ``[1, 60]``; invalid
        values are replaced with the default of 5.
    conf:
        Minimum confidence threshold for detections.  Must be in
        ``(0, 1]``; invalid values are replaced with the default of 0.25.
    """
    # ── Validate parameters ──────────────────────────────────────────────
    skip = _validate_skip(skip)
    conf = _validate_conf(conf)

    # ── Load slot configuration ──────────────────────────────────────────
    cfg = load_slots(source)
    if not cfg.slots:
        _log.warning(
            "No slots defined for source '%s'. "
            "Run: python mark_parking_slots.py --source %s",
            source.id,
            source.id,
        )

    n_slots = len(cfg.slots)

    # ── Resolve homography ───────────────────────────────────────────────
    H: Optional[np.ndarray] = None
    warped_size: Optional[Tuple[int, int]] = None

    if cfg.homography_matrix is not None:
        H = np.array(cfg.homography_matrix, dtype=np.float64)
        if cfg.warped_size is not None:
            warped_size = (int(cfg.warped_size[0]), int(cfg.warped_size[1]))
        else:
            _log.warning(
                "homography_matrix is set but warped_size is None for source '%s'. "
                "Using original frame size.",
                source.id,
            )

    # ── Load YOLO model ──────────────────────────────────────────────────
    model = load_model(model_path)

    # ── Open video capture ───────────────────────────────────────────────
    cap = cv2.VideoCapture(source.uri)
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(source.uri))
        except (ValueError, TypeError):
            pass
    if not cap.isOpened():
        _log.error("Cannot open source: %s", source.uri)
        return

    is_file = not (
        source.uri.startswith("rtsp")
        or source.uri.startswith("rtmp")
        or (source.uri.isdigit())
    )

    entry: Tuple[int, int] = tuple(source.entry_point)  # type: ignore[assignment]

    # ── Initialise Frame_Cache ───────────────────────────────────────────
    cache = FrameCache.make(n_slots)

    # ── A* cache — recompute only when statuses change ───────────────────
    cached_statuses: Optional[List[bool]] = None
    cached_path: Optional[List[Tuple[int, int]]] = None
    cached_nearest: Optional[int] = None

    frame_index: int = 0
    t_prev: float = time.time()

    _log.info(
        "Running YOLO pipeline for '%s'  (skip=%d, conf=%.2f)  — press q to quit",
        source.name,
        skip,
        conf,
    )

    while True:
        ret, raw_frame = cap.read()

        # ── Handle end-of-stream ─────────────────────────────────────────
        if not ret:
            if is_file:
                # Loop video file back to the beginning (Requirement 3.x)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                continue
            else:
                # RTSP: attempt reconnect with 1 s back-off
                _log.error(
                    "Stream ended for source '%s'. Attempting reconnect…",
                    source.name,
                )
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(source.uri)
                if not cap.isOpened():
                    _log.error(
                        "Reconnect failed for source '%s'. Exiting.",
                        source.name,
                    )
                    break
                continue

        # ── Apply crop region (if set) ───────────────────────────────────
        # All slot coords are relative to the cropped frame.
        # Do this BEFORE YOLO inference so detections align with slots.
        raw_frame_cropped = apply_crop(raw_frame, cfg)

        # ── FIX 1: detect on raw_frame_cropped (side-view), not warped ──
        if frame_index % skip == 0:
            try:
                raw_dets = model(raw_frame_cropped, verbose=False)[0]
                filtered_original = filter_detections(raw_dets, conf)

                # Project detections into warped coordinate space
                if H is not None:
                    filtered_warped = project_boxes_to_warped(filtered_original, H)
                else:
                    filtered_warped = filtered_original

                cache.statuses = assign_slots(filtered_warped, cfg)
                cache.boxes_original = filtered_original
                cache.boxes = filtered_warped
                cache.is_initialized = True

                # ── DEBUG WINDOW: show ALL detections on cropped frame ───
                debug_frame = raw_frame_cropped.copy()
                all_boxes = raw_dets.boxes
                if all_boxes is not None:
                    for box in all_boxes:
                        cls_id = int(box.cls[0])
                        cls_name = raw_dets.names[cls_id]
                        c = float(box.conf[0])
                        dx1, dy1, dx2, dy2 = (int(v) for v in box.xyxy[0])
                        color = (0, 255, 0) if cls_name in VEHICLE_CLASSES else (0, 0, 255)
                        cv2.rectangle(debug_frame, (dx1, dy1), (dx2, dy2), color, 2)
                        cv2.putText(
                            debug_frame,
                            f"{cls_name} {c:.2f}",
                            (dx1, max(dy1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        )
                        # Bottom-center dot on original frame
                        bx_orig, by_orig = bottom_center(dx1, dy1, dx2, dy2)
                        cv2.circle(debug_frame, (bx_orig, by_orig), 5, (255, 0, 255), -1)
                cv2.putText(
                    debug_frame,
                    f"ALL detections (conf>0) — green=vehicle, red=other",
                    (10, debug_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                )
                cv2.imshow("DEBUG — YOLO on original frame", debug_frame)

            except Exception:
                _log.exception(
                    "YOLO inference failed on frame %d; using last cached result.",
                    frame_index,
                )
                filtered_original = getattr(cache, "boxes_original", [])
                filtered_warped = cache.boxes
            overlay_text = "LIVE"
        else:
            filtered_original = getattr(cache, "boxes_original", [])
            filtered_warped = cache.boxes
            overlay_text = "CACHED"

        # ── Warp cropped frame for display only (after detection) ───────
        if H is not None and warped_size is not None:
            frame = warp_frame(raw_frame_cropped, H, warped_size)
        else:
            frame = raw_frame_cropped.copy()

        # ── Slot statuses: FrameCache uses True=occupied; utils expects True=free
        free_statuses: List[bool] = [not s for s in cache.statuses]
        free_count: int = sum(free_statuses)

        # ── A* pathfinding (recompute only when statuses change) ─────────
        if free_statuses != cached_statuses:
            cached_statuses = free_statuses[:]
            nearest_idx = find_nearest_free_slot(entry, cfg, free_statuses)
            cached_nearest = nearest_idx

            if nearest_idx is not None:
                n_rect = len(cfg.slots)
                if nearest_idx < n_rect:
                    # Rectangle slot — use rect center
                    gx = cfg.slots[nearest_idx][0] + cfg.rect_w // 2
                    gy = cfg.slots[nearest_idx][1] + cfg.rect_h // 2
                else:
                    # Polygon slot — use centroid
                    poly = cfg.poly_slots[nearest_idx - n_rect]
                    pts = np.array(poly, dtype=np.float32)
                    gx = int(pts[:, 0].mean())
                    gy = int(pts[:, 1].mean())
                obstacle_grid = build_obstacle_grid(frame.shape, cfg, free_statuses)
                cached_path = astar(obstacle_grid, entry, (gx, gy))
            else:
                cached_path = None

        # ── Draw projected center + bottom-center dots on warped frame ──
        # Cyan = bbox center (used for polygon slots)
        # Magenta = bottom-center (used for rect slots)
        for cx, cy, bx, by, class_name, confidence in filtered_warped:
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)   # cyan = center
            cv2.circle(frame, (bx, by), 5, (255, 0, 255), -1)   # magenta = bottom-ctr

        # ── Draw slots, path, HUD, and LIVE/CACHED overlay ───────────────
        out = draw_slots(
            frame,
            cfg,
            free_statuses,
            free_count,
            cached_nearest,
            cached_path,
            entry,
            source.name,
            extra_overlay=overlay_text,
        )

        # ── FPS display ───────────────────────────────────────────────────
        t_now = time.time()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        cv2.putText(
            out,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 110, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
        )

        cv2.imshow(f"YOLO Parking Counter — {source.name}", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO-based parking space counter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python detect_yolo.py --list\n"
            "  python detect_yolo.py --source carpark_main\n"
            "  python detect_yolo.py --source carpark_main --model yolov8s.pt "
            "--skip 3 --conf 0.4\n"
        ),
    )
    parser.add_argument(
        "--source",
        metavar="ID",
        help="Source ID from sources.json (default: first source)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available sources and exit",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        default="yolov8n.pt",
        help="YOLO model path or name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--skip",
        metavar="N",
        type=int,
        default=_DEFAULT_SKIP,
        help=f"Run inference every N frames, range [1,60] (default: {_DEFAULT_SKIP})",
    )
    parser.add_argument(
        "--conf",
        metavar="F",
        type=float,
        default=_DEFAULT_CONF,
        help=f"Minimum detection confidence, range (0,1] (default: {_DEFAULT_CONF})",
    )

    args = parser.parse_args()

    if args.list:
        list_sources()
        sys.exit(0)

    sources = load_sources()
    if not sources:
        _log.error("No sources defined in sources.json. Run: python config.py --add")
        sys.exit(1)

    if args.source:
        source = get_source_by_id(args.source)
        if source is None:
            _log.error("Source '%s' not found.", args.source)
            list_sources()
            sys.exit(1)
    elif len(sources) == 1:
        source = sources[0]
    else:
        list_sources()
        print()
        chosen = input("Enter source ID to run: ").strip()
        source = get_source_by_id(chosen)
        if source is None:
            _log.error("Source '%s' not found.", chosen)
            sys.exit(1)

    run(source, model_path=args.model, skip=args.skip, conf=args.conf)

"""
app.py
======
FastAPI web GUI for the parking space counter system.

Endpoints (tasks 8.2 and 8.3 — NOT yet implemented here):
    GET  /                          HTML page with video + controls
    GET  /stream/{source_id}        MJPEG stream
    GET  /status/{source_id}        JSON status
    POST /switch/{source_id}        Toggle Classical ↔ YOLO
    GET  /sources                   List available sources

This module (task 8.1) implements:
    - PipelineState dataclass
    - Pipeline thread functions (classical and YOLO inline loops)
    - create_pipeline_thread()
    - switch_pipeline()
    - FastAPI startup event that loads sources and initialises pipeline states
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from config import Source, SlotConfig, load_slots, load_sources
from detect_improved import convert_grayscale, check_slots
from detect_yolo import (
    FrameCache,
    assign_slots,
    filter_detections,
    load_model,
    warp_frame,
)
from utils import (
    astar,
    apply_crop,
    build_obstacle_grid,
    draw_slots,
    find_nearest_free_slot,
)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# URI resolution helper
# ─────────────────────────────────────────────

# Resolve relative URIs relative to the directory containing app.py,
# not the shell working directory. This ensures ./video/foo.mp4 works
# regardless of where uvicorn is launched from.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_uri(uri: str) -> str:
    """Return an absolute path for file URIs; leave RTSP/device URIs unchanged."""
    if uri.startswith("rtsp") or uri.startswith("rtmp") or uri.isdigit():
        return uri
    if os.path.isabs(uri):
        return uri
    return os.path.normpath(os.path.join(_PROJECT_DIR, uri))

# ─────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────

app = FastAPI(title="Parking Space Counter")

# ─────────────────────────────────────────────
# PipelineState dataclass
# ─────────────────────────────────────────────


@dataclass
class PipelineState:
    """Holds all runtime state for one active video source.

    Fields
    ------
    source       : The video/RTSP source being processed.
    mode         : Active detection mode — ``"classical"`` or ``"yolo"``.
    thread       : Background processing thread.
    stop_event   : Set to signal the thread to stop.
    latest_frame : Most recently annotated frame (protected by ``frame_lock``).
    frame_lock   : Mutex for ``latest_frame``.
    status       : Dict with keys ``free_slots``, ``total_slots``,
                   ``nearest_free_slot``, ``fps``, ``mode``, ``source_name``
                   (protected by ``status_lock``).
    status_lock  : Mutex for ``status``.
    """

    source: Source
    mode: str  # "classical" | "yolo"
    thread: threading.Thread
    stop_event: threading.Event
    latest_frame: Optional[np.ndarray]  # protected by frame_lock
    frame_lock: threading.Lock
    status: dict  # free_slots, total_slots, nearest_free_slot, fps, mode
    status_lock: threading.Lock


# ─────────────────────────────────────────────
# Global pipeline state registry
# ─────────────────────────────────────────────

#: Keyed by source_id → PipelineState
pipeline_states: Dict[str, PipelineState] = {}

# ─────────────────────────────────────────────
# Default YOLO parameters (used by the inline YOLO loop)
# ─────────────────────────────────────────────

_DEFAULT_MODEL_PATH: str = "yolov8n.pt"
_DEFAULT_SKIP: int = 5
_DEFAULT_CONF: float = 0.25

# ─────────────────────────────────────────────
# Pipeline thread functions
# ─────────────────────────────────────────────


def _classical_pipeline_loop(state: PipelineState) -> None:
    """Inline classical CV pipeline loop.

    Opens the video capture for ``state.source``, reads frames, applies the
    classical grayscale/threshold/pixel-count detection, updates
    ``state.latest_frame`` and ``state.status``, and exits when
    ``state.stop_event`` is set.
    """
    source = state.source
    cfg: SlotConfig = load_slots(source)

    if not cfg.slots and not cfg.poly_slots:
        _log.warning(
            "[classical] No slots defined for '%s'. "
            "Run: python mark_parking_slots.py --source %s",
            source.id,
            source.id,
        )

    # ── Resolve homography (same as YOLO pipeline) ───────────────────────
    H_cl: Optional[np.ndarray] = None
    warped_size_cl: Optional[tuple] = None
    if cfg.homography_matrix is not None:
        H_cl = np.array(cfg.homography_matrix, dtype=np.float64)
        if cfg.warped_size is not None:
            warped_size_cl = (int(cfg.warped_size[0]), int(cfg.warped_size[1]))

    cap = cv2.VideoCapture(_resolve_uri(source.uri))
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(source.uri))
        except (ValueError, TypeError):
            pass
    if not cap.isOpened():
        _log.error("[classical] Cannot open source: %s", source.uri)
        with state.status_lock:
            state.status["error"] = f"Cannot open source: {source.uri}"
        return

    is_file = not (
        source.uri.startswith("rtsp")
        or source.uri.startswith("rtmp")
        or source.uri.isdigit()
    )

    entry: Tuple[int, int] = tuple(source.entry_point)  # type: ignore[assignment]

    # A* cache — recompute only when slot statuses change
    cached_statuses: Optional[List[bool]] = None
    cached_path: Optional[List[Tuple[int, int]]] = None
    cached_nearest: Optional[int] = None

    t_prev = time.time()

    _log.info("[classical] Pipeline started for '%s'", source.name)

    while not state.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                _log.error(
                    "[classical] Stream ended for '%s'. Attempting reconnect…",
                    source.name,
                )
                cap.release()
                time.sleep(1.0)
                if state.stop_event.is_set():
                    break
                cap = cv2.VideoCapture(_resolve_uri(source.uri))
                if not cap.isOpened():
                    _log.error(
                        "[classical] Reconnect failed for '%s'. Exiting.",
                        source.name,
                    )
                    break
                continue

        cropped = apply_crop(frame, cfg)
        # Apply homography warp — slot coords are in warped space
        if H_cl is not None and warped_size_cl is not None:
            working = cv2.warpPerspective(cropped, H_cl, warped_size_cl)
        else:
            working = cropped
        gray_frame = convert_grayscale(working)
        statuses, free_count = check_slots(gray_frame, cfg)

        if statuses != cached_statuses:
            cached_statuses = statuses[:]
            nearest_idx = find_nearest_free_slot(entry, cfg, statuses)
            cached_nearest = nearest_idx

            if nearest_idx is not None:
                n_rect = len(cfg.slots)
                if nearest_idx < n_rect:
                    gx = cfg.slots[nearest_idx][0] + cfg.rect_w // 2
                    gy = cfg.slots[nearest_idx][1] + cfg.rect_h // 2
                else:
                    poly = cfg.poly_slots[nearest_idx - n_rect]
                    pts = np.array(poly, dtype=np.float32)
                    gx = int(pts[:, 0].mean())
                    gy = int(pts[:, 1].mean())
                obstacle_grid = build_obstacle_grid(working.shape, cfg, statuses)
                cached_path = astar(obstacle_grid, entry, (gx, gy))
            else:
                cached_path = None

        out = draw_slots(
            working,
            cfg,
            statuses,
            free_count,
            cached_nearest,
            cached_path,
            entry,
            source.name,
        )

        t_now = time.time()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        cv2.putText(
            out,
            f"FPS: {fps:.1f}",
            (out.shape[1] - 110, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
        )

        # Update shared state (protected by locks)
        with state.frame_lock:
            state.latest_frame = out.copy()

        nearest_display = (cached_nearest + 1) if cached_nearest is not None else None
        with state.status_lock:
            state.status.update(
                {
                    "free_slots": free_count,
                    "total_slots": len(cfg.slots) + len(cfg.poly_slots),
                    "nearest_free_slot": nearest_display,
                    "fps": round(fps, 1),
                    "mode": "classical",
                    "source_name": source.name,
                }
            )

    cap.release()
    _log.info("[classical] Pipeline stopped for '%s'", source.name)


def _yolo_pipeline_loop(
    state: PipelineState,
    model_path: str = _DEFAULT_MODEL_PATH,
    skip: int = _DEFAULT_SKIP,
    conf: float = _DEFAULT_CONF,
) -> None:
    """Inline YOLO pipeline loop.

    Opens the video capture for ``state.source``, applies perspective warp
    when a homography matrix is available, runs YOLO inference every *skip*
    frames (using the Frame_Cache otherwise), updates ``state.latest_frame``
    and ``state.status``, and exits when ``state.stop_event`` is set.
    """
    source = state.source
    cfg: SlotConfig = load_slots(source)

    if not cfg.slots:
        _log.warning(
            "[yolo] No slots defined for '%s'. "
            "Run: python mark_parking_slots.py --source %s",
            source.id,
            source.id,
        )

    n_slots = len(cfg.slots)

    # Resolve homography
    H: Optional[np.ndarray] = None
    warped_size: Optional[Tuple[int, int]] = None

    if cfg.homography_matrix is not None:
        H = np.array(cfg.homography_matrix, dtype=np.float64)
        if cfg.warped_size is not None:
            warped_size = (int(cfg.warped_size[0]), int(cfg.warped_size[1]))
        else:
            _log.warning(
                "[yolo] homography_matrix set but warped_size is None for '%s'. "
                "Using original frame size.",
                source.id,
            )

    # Load YOLO model
    try:
        model = load_model(model_path)
    except SystemExit:
        _log.error("[yolo] Failed to load model '%s'. Exiting thread.", model_path)
        with state.status_lock:
            state.status["error"] = f"Model not found: {model_path}"
        return

    # Open video capture
    cap = cv2.VideoCapture(_resolve_uri(source.uri))
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(source.uri))
        except (ValueError, TypeError):
            pass
    if not cap.isOpened():
        _log.error("[yolo] Cannot open source: %s", source.uri)
        with state.status_lock:
            state.status["error"] = f"Cannot open source: {source.uri}"
        return

    is_file = not (
        source.uri.startswith("rtsp")
        or source.uri.startswith("rtmp")
        or source.uri.isdigit()
    )

    entry: Tuple[int, int] = tuple(source.entry_point)  # type: ignore[assignment]

    cache = FrameCache.make(n_slots)

    # A* cache
    cached_statuses: Optional[List[bool]] = None
    cached_path: Optional[List[Tuple[int, int]]] = None
    cached_nearest: Optional[int] = None

    frame_index: int = 0
    t_prev: float = time.time()

    _log.info(
        "[yolo] Pipeline started for '%s' (skip=%d, conf=%.2f)",
        source.name,
        skip,
        conf,
    )

    while not state.stop_event.is_set():
        ret, raw_frame = cap.read()
        if not ret:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                continue
            else:
                _log.error(
                    "[yolo] Stream ended for '%s'. Attempting reconnect…",
                    source.name,
                )
                cap.release()
                time.sleep(1.0)
                if state.stop_event.is_set():
                    break
                cap = cv2.VideoCapture(_resolve_uri(source.uri))
                if not cap.isOpened():
                    _log.error(
                        "[yolo] Reconnect failed for '%s'. Exiting.",
                        source.name,
                    )
                    break
                continue

        # Apply crop then warp for display
        raw_cropped = apply_crop(raw_frame, cfg)

        # Perspective warp
        if H is not None and warped_size is not None:
            frame = warp_frame(raw_cropped, H, warped_size)
        else:
            frame = raw_cropped.copy()

        # Inference or cache (run on cropped frame)
        if frame_index % skip == 0:
            try:
                raw_dets = model(raw_cropped, verbose=False)[0]
                filtered = filter_detections(raw_dets, conf)
                # Project detections into warped coordinate space for slot assignment
                from detect_yolo import project_boxes_to_warped as _project
                filtered_for_slots = _project(filtered, H) if H is not None else filtered
                cache.statuses = assign_slots(filtered_for_slots, cfg)
                cache.boxes = filtered        # keep original for bbox drawing
                cache.is_initialized = True
            except Exception:
                _log.exception(
                    "[yolo] Inference failed on frame %d; using last cached result.",
                    frame_index,
                )
                filtered = cache.boxes
            overlay_text = "LIVE"
        else:
            filtered = cache.boxes
            overlay_text = "CACHED"

        # FrameCache uses True=occupied; utils expects True=free — invert
        free_statuses: List[bool] = [not s for s in cache.statuses]
        free_count: int = sum(free_statuses)

        # A* pathfinding (recompute only when statuses change)
        if free_statuses != cached_statuses:
            cached_statuses = free_statuses[:]
            nearest_idx = find_nearest_free_slot(entry, cfg, free_statuses)
            cached_nearest = nearest_idx

            if nearest_idx is not None:
                n_rect = len(cfg.slots)
                if nearest_idx < n_rect:
                    gx = cfg.slots[nearest_idx][0] + cfg.rect_w // 2
                    gy = cfg.slots[nearest_idx][1] + cfg.rect_h // 2
                else:
                    poly = cfg.poly_slots[nearest_idx - n_rect]
                    pts = np.array(poly, dtype=np.float32)
                    gx = int(pts[:, 0].mean())
                    gy = int(pts[:, 1].mean())
                obstacle_grid = build_obstacle_grid(frame.shape, cfg, free_statuses)
                cached_path = astar(obstacle_grid, entry, (gx, gy))
            else:
                cached_path = None

        # Draw bounding boxes (cyan) and bottom-center points (magenta)
        from detect_yolo import bottom_center  # local import to avoid circular issues

        for x1, y1, x2, y2, class_name, confidence in filtered:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )
            bx, by = bottom_center(x1, y1, x2, y2)
            cv2.circle(frame, (bx, by), 5, (255, 0, 255), -1)

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

        # Update shared state (protected by locks)
        with state.frame_lock:
            state.latest_frame = out.copy()

        nearest_display = (cached_nearest + 1) if cached_nearest is not None else None
        with state.status_lock:
            state.status.update(
                {
                    "free_slots": free_count,
                    "total_slots": n_slots + len(cfg.poly_slots),
                    "nearest_free_slot": nearest_display,
                    "fps": round(fps, 1),
                    "mode": "yolo",
                    "source_name": source.name,
                    "skip": skip,
                }
            )

        frame_index += 1

    cap.release()
    _log.info("[yolo] Pipeline stopped for '%s'", source.name)


# ─────────────────────────────────────────────
# Thread factory
# ─────────────────────────────────────────────


def create_pipeline_thread(state: PipelineState) -> threading.Thread:
    """Create (but do not start) a daemon thread for the given pipeline state.

    The thread target is chosen based on ``state.mode``:
    - ``"classical"`` → :func:`_classical_pipeline_loop`
    - ``"yolo"``      → :func:`_yolo_pipeline_loop`

    Parameters
    ----------
    state:
        The :class:`PipelineState` whose ``mode`` determines which pipeline
        loop to run.

    Returns
    -------
    threading.Thread
        A new, not-yet-started daemon thread.
    """
    if state.mode == "yolo":
        target = _yolo_pipeline_loop
        args = (state,)
    else:
        target = _classical_pipeline_loop
        args = (state,)

    t = threading.Thread(target=target, args=args, daemon=True)
    return t


# ─────────────────────────────────────────────
# Mode switching
# ─────────────────────────────────────────────


def switch_pipeline(state: PipelineState, new_mode: str) -> None:
    """Switch the active pipeline for *state* to *new_mode*.

    Implements the mode-switch sequence (Requirement 5.8):

    1. Set ``stop_event`` to signal the current thread to stop.
    2. Join the current thread with a 3-second timeout.
    3. Update ``state.mode`` to *new_mode*.
    4. Create a new thread via :func:`create_pipeline_thread`.
    5. Clear ``stop_event`` and start the new thread.

    During the transition ``state.latest_frame`` is preserved so that the
    ``/stream`` endpoint can continue serving the last available frame.

    Parameters
    ----------
    state:
        The :class:`PipelineState` to switch.
    new_mode:
        ``"classical"`` or ``"yolo"``.
    """
    if new_mode not in ("classical", "yolo"):
        _log.error("switch_pipeline: unknown mode '%s'", new_mode)
        return

    if state.mode == new_mode:
        _log.info(
            "switch_pipeline: source '%s' is already in '%s' mode — no-op.",
            state.source.id,
            new_mode,
        )
        return

    _log.info(
        "Switching source '%s' from '%s' to '%s'…",
        state.source.id,
        state.mode,
        new_mode,
    )

    # Step 1: signal the current thread to stop
    state.stop_event.set()

    # Step 2: join with 3-second timeout (Requirement 5.8)
    state.thread.join(timeout=3.0)
    if state.thread.is_alive():
        _log.warning(
            "Pipeline thread for '%s' did not stop within 3 s; "
            "proceeding with new thread anyway.",
            state.source.id,
        )

    # Step 3: update mode
    state.mode = new_mode

    # Step 4: create new thread
    state.stop_event.clear()
    new_thread = create_pipeline_thread(state)

    # Step 5: start new thread
    state.thread = new_thread
    state.thread.start()

    _log.info(
        "Source '%s' switched to '%s' mode.",
        state.source.id,
        new_mode,
    )


# ─────────────────────────────────────────────
# Startup event
# ─────────────────────────────────────────────


@app.on_event("startup")
async def startup_event() -> None:
    """Load sources and initialise pipeline states on server startup.

    Requirement 5.2: if ``sources.json`` is missing or contains no sources,
    log an error and exit with status code 1.
    """
    sources = load_sources()

    if not sources:
        _log.error(
            "sources.json is missing or contains no sources. "
            "Add at least one source via: python config.py --add"
        )
        sys.exit(1)

    _log.info("Loaded %d source(s) from sources.json.", len(sources))

    for source in sources:
        stop_event = threading.Event()
        initial_status: dict = {
            "free_slots": 0,
            "total_slots": 0,
            "nearest_free_slot": None,
            "fps": 0.0,
            "mode": "classical",
            "source_name": source.name,
        }

        state = PipelineState(
            source=source,
            mode="classical",
            thread=threading.Thread(),  # placeholder; replaced below
            stop_event=stop_event,
            latest_frame=None,
            frame_lock=threading.Lock(),
            status=initial_status,
            status_lock=threading.Lock(),
        )

        # Create and start the initial classical pipeline thread
        thread = create_pipeline_thread(state)
        state.thread = thread
        thread.start()

        pipeline_states[source.id] = state
        _log.info("Started classical pipeline for source '%s'.", source.id)


# ─────────────────────────────────────────────
# MJPEG stream helpers
# ─────────────────────────────────────────────

_PLACEHOLDER_FRAME: Optional[bytes] = None


def _get_placeholder_jpeg() -> bytes:
    """Return a 320×240 black JPEG used when no frame is available yet."""
    global _PLACEHOLDER_FRAME
    if _PLACEHOLDER_FRAME is None:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank)
        _PLACEHOLDER_FRAME = buf.tobytes()
    return _PLACEHOLDER_FRAME


async def _frame_generator(source_id: str) -> AsyncGenerator[bytes, None]:
    """Async generator that yields MJPEG boundary-delimited JPEG frames."""
    state = pipeline_states[source_id]
    while True:
        with state.frame_lock:
            frame = state.latest_frame

        if frame is None:
            jpeg_bytes = _get_placeholder_jpeg()
        else:
            _, buf = cv2.imencode(".jpg", frame)
            jpeg_bytes = buf.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )
        await asyncio.sleep(0.033)  # ~30 fps cap


# ─────────────────────────────────────────────
# Endpoints — Task 8.2
# ─────────────────────────────────────────────


@app.get("/stream/{source_id}")
async def stream(source_id: str) -> StreamingResponse:
    """MJPEG stream for the given source.

    Returns HTTP 404 when *source_id* is not registered.
    """
    if source_id not in pipeline_states:
        raise HTTPException(status_code=404, detail=f"Unknown source: {source_id}")
    return StreamingResponse(
        _frame_generator(source_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Self-contained HTML page with MJPEG viewer, source selector, mode switch, and HUD."""
    # Build source options for the <select>
    source_options = "\n".join(
        f'<option value="{sid}">{state.source.name}</option>'
        for sid, state in pipeline_states.items()
    )

    # Pick the first source as the default
    default_source = next(iter(pipeline_states), "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Parking Space Counter</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: Arial, sans-serif;
      background: #1a1a2e;
      color: #e0e0e0;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      gap: 16px;
    }}
    h1 {{ font-size: 1.6rem; color: #a0c4ff; }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: center;
    }}
    select, button {{
      padding: 8px 14px;
      border-radius: 6px;
      border: none;
      font-size: 0.95rem;
      cursor: pointer;
    }}
    select {{ background: #2d2d44; color: #e0e0e0; }}
    button {{
      background: #4361ee;
      color: #fff;
      transition: background 0.2s;
    }}
    button:hover {{ background: #3a0ca3; }}
    #stream-img {{
      max-width: 100%;
      border: 2px solid #4361ee;
      border-radius: 8px;
    }}
    .hud {{
      background: #16213e;
      border: 1px solid #4361ee;
      border-radius: 8px;
      padding: 14px 20px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      width: 100%;
      max-width: 720px;
    }}
    .hud-item {{ display: flex; flex-direction: column; gap: 2px; }}
    .hud-label {{ font-size: 0.75rem; color: #888; text-transform: uppercase; }}
    .hud-value {{ font-size: 1.1rem; font-weight: bold; color: #a0c4ff; }}
    #error-msg {{
      color: #ff6b6b;
      font-size: 0.9rem;
      display: none;
    }}
  </style>
</head>
<body>
  <h1>🅿 Parking Space Counter</h1>

  <div class="controls">
    <label for="source-select" style="font-size:0.9rem;">Source:</label>
    <select id="source-select" onchange="changeSource(this.value)">
      {source_options}
    </select>
    <button id="mode-btn" onclick="switchMode()">Switch Mode</button>
  </div>

  <p id="error-msg"></p>

  <img id="stream-img" src="/stream/{default_source}" alt="MJPEG stream"
       onerror="showError('Stream unavailable for this source.')" />

  <div class="hud">
    <div class="hud-item">
      <span class="hud-label">Free Slots</span>
      <span class="hud-value" id="hud-free">—</span>
    </div>
    <div class="hud-item">
      <span class="hud-label">Total Slots</span>
      <span class="hud-value" id="hud-total">—</span>
    </div>
    <div class="hud-item">
      <span class="hud-label">Nearest Free</span>
      <span class="hud-value" id="hud-nearest">—</span>
    </div>
    <div class="hud-item">
      <span class="hud-label">FPS</span>
      <span class="hud-value" id="hud-fps">—</span>
    </div>
    <div class="hud-item">
      <span class="hud-label">Mode</span>
      <span class="hud-value" id="hud-mode">—</span>
    </div>
    <div class="hud-item" id="hud-skip-item" style="display:none;">
      <span class="hud-label">Frame Skip Interval</span>
      <span class="hud-value" id="hud-skip">5</span>
    </div>
  </div>

  <script>
    let currentSource = "{default_source}";

    function showError(msg) {{
      const el = document.getElementById("error-msg");
      el.textContent = msg;
      el.style.display = "block";
    }}

    function hideError() {{
      document.getElementById("error-msg").style.display = "none";
    }}

    function changeSource(sourceId) {{
      currentSource = sourceId;
      hideError();
      const img = document.getElementById("stream-img");
      img.src = "/stream/" + sourceId;
      img.onerror = () => showError("Stream unavailable for source: " + sourceId);
      updateHUD();
    }}

    async function switchMode() {{
      try {{
        const resp = await fetch("/switch/" + currentSource, {{ method: "POST" }});
        if (!resp.ok) {{
          const data = await resp.json();
          showError("Switch failed: " + (data.detail || resp.status));
          return;
        }}
        const data = await resp.json();
        document.getElementById("hud-mode").textContent = data.mode;
      }} catch (e) {{
        showError("Switch request failed: " + e.message);
      }}
    }}

    async function updateHUD() {{
      try {{
        const resp = await fetch("/status/" + currentSource);
        if (!resp.ok) {{
          showError("Status unavailable (HTTP " + resp.status + ")");
          return;
        }}
        const d = await resp.json();
        document.getElementById("hud-free").textContent    = d.free_slots ?? "—";
        document.getElementById("hud-total").textContent   = d.total_slots ?? "—";
        document.getElementById("hud-nearest").textContent = d.nearest_free_slot ?? "None";
        document.getElementById("hud-fps").textContent     = d.fps != null ? d.fps.toFixed(1) : "—";
        document.getElementById("hud-mode").textContent    = d.mode ?? "—";

        const skipItem = document.getElementById("hud-skip-item");
        if (d.mode === "yolo") {{
          skipItem.style.display = "";
          document.getElementById("hud-skip").textContent = d.skip ?? 5;
        }} else {{
          skipItem.style.display = "none";
        }}
        hideError();
      }} catch (e) {{
        showError("HUD update failed: " + e.message);
      }}
    }}

    // Poll status every second
    setInterval(updateHUD, 1000);
    updateHUD();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────
# Endpoints — Task 8.3
# ─────────────────────────────────────────────


@app.get("/status/{source_id}")
async def status(source_id: str) -> JSONResponse:
    """Return JSON status for the given source.

    Returns HTTP 404 when *source_id* is not registered.
    """
    if source_id not in pipeline_states:
        raise HTTPException(status_code=404, detail=f"Unknown source: {source_id}")
    state = pipeline_states[source_id]
    with state.status_lock:
        data = dict(state.status)
    # Ensure all required keys are present (defensive defaults)
    response = {
        "free_slots": data.get("free_slots", 0),
        "total_slots": data.get("total_slots", 0),
        "nearest_free_slot": data.get("nearest_free_slot", None),
        "fps": data.get("fps", 0.0),
        "mode": data.get("mode", state.mode),
        "source_name": data.get("source_name", state.source.name),
    }
    return JSONResponse(content=response)


@app.post("/switch/{source_id}")
async def switch(source_id: str) -> JSONResponse:
    """Toggle the pipeline mode (Classical ↔ YOLO) for the given source.

    Returns HTTP 404 when *source_id* is not registered.
    """
    if source_id not in pipeline_states:
        raise HTTPException(status_code=404, detail=f"Unknown source: {source_id}")
    state = pipeline_states[source_id]
    new_mode = "yolo" if state.mode == "classical" else "classical"
    switch_pipeline(state, new_mode)
    return JSONResponse(content={"status": "ok", "mode": new_mode})


@app.get("/sources")
async def sources() -> JSONResponse:
    """Return a list of all registered sources."""
    result = [
        {"id": sid, "name": state.source.name}
        for sid, state in pipeline_states.items()
    ]
    return JSONResponse(content=result)


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Parking Space Counter web server")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port to listen on (default: 8000)",
    )
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)

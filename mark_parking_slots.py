"""
mark_parking_slots.py
=====================
Interactive tool to mark parking slot positions for a given source.

Usage
-----
  python mark_parking_slots.py                     # picks first source
  python mark_parking_slots.py --source carpark_main
  python mark_parking_slots.py --list              # list available sources

Controls
--------
  Drag left-click  → draw first rectangle (sets default size)
  Left-click       → place rectangle with saved size
  Right-click      → remove last rectangle
  s                → save slots to file
  r                → clear all slots
  c                → reset default size (drag again)
  n / p            → next / previous frame
  q                → quit (auto-saves if slots exist)
"""

from __future__ import annotations

import sys
import cv2
import numpy as np
from typing import List, Optional, Tuple

from config import (
    load_sources, load_slots, save_slots, save_sources,
    get_source_by_id, list_sources,
    Source, SlotConfig,
)

# ─────────────────────────────────────────────
# Global UI state
# ─────────────────────────────────────────────
parking_slots: list  = []   # [(x, y, w, h), ...]
current_frame        = None
drawing              = False
start_point          = None
temp_rect            = None
saved_width          = None
saved_height         = None
size_saved           = False
window_name          = ""


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────

def draw_rectangles(frame, slots, temp=None):
    for idx, (x, y, w, h) in enumerate(slots):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(idx + 1), (x - 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{w}x{h}", (x + 5, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if temp is not None:
        x, y, w, h = temp
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{w}x{h}", (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if size_saved and saved_width:
        info = f"Default size: {saved_width}x{saved_height}"
        tw = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv2.putText(frame, info, (frame.shape[1] - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        info = "No default size — drag to set"
        tw = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
        cv2.putText(frame, info, (frame.shape[1] - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    cv2.putText(frame, f"Slots: {len(slots)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    return frame


# ─────────────────────────────────────────────
# Mouse callback
# ─────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    global parking_slots, current_frame, drawing, start_point, temp_rect
    global saved_width, saved_height, size_saved

    if event == cv2.EVENT_LBUTTONDOWN:
        if size_saved and saved_width:
            parking_slots.append((x, y, saved_width, saved_height))
            print(f"  + Slot #{len(parking_slots)}: ({x}, {y})  {saved_width}x{saved_height}")
            _refresh()
        else:
            drawing = True
            start_point = (x, y)
            temp_rect = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if not size_saved and drawing and start_point:
            x1, y1 = start_point
            w, h = abs(x - x1), abs(y - y1)
            temp_rect = (min(x1, x), min(y1, y), w, h)
            _refresh(temp_rect)

    elif event == cv2.EVENT_LBUTTONUP:
        if not size_saved and drawing and start_point:
            x1, y1 = start_point
            w, h = abs(x - x1), abs(y - y1)
            if w >= 10 and h >= 10:
                xs, ys = min(x1, x), min(y1, y)
                parking_slots.append((xs, ys, w, h))
                saved_width, saved_height, size_saved = w, h, True
                print(f"  + Slot #{len(parking_slots)}: ({xs}, {ys})  {w}x{h}  [size locked]")
            else:
                print(f"  ! Too small ({w}x{h}), ignored")
            drawing = False
            start_point = None
            temp_rect = None
            _refresh()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if parking_slots:
            removed = parking_slots.pop()
            print(f"  - Removed slot: ({removed[0]}, {removed[1]})")
            _refresh()


def _refresh(temp=None):
    if current_frame is not None:
        fc = draw_rectangles(current_frame.copy(), parking_slots, temp)
        cv2.imshow(window_name, fc)


# ─────────────────────────────────────────────
# Homography calibration helpers
# ─────────────────────────────────────────────

def is_collinear(pts: List[Tuple[int, int]], tol: float = 1e-6) -> bool:
    """Return True if the 4 points are collinear / nearly collinear (degenerate).

    Checks whether any three consecutive points (in the quadrilateral formed by
    pts[0]→pts[1]→pts[2]→pts[3]) are collinear by computing the magnitude of
    the cross product of consecutive edge vectors.  If *any* cross product
    magnitude is below *tol* the quad is considered degenerate.

    Requirements: 1.8
    """
    if len(pts) != 4:
        raise ValueError(f"is_collinear expects exactly 4 points, got {len(pts)}")

    # Build edge vectors for the closed polygon (4 edges)
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cx, cy = pts[(i + 2) % n]
        # Edge vectors AB and BC
        abx, aby = bx - ax, by - ay
        bcx, bcy = cx - bx, cy - by
        # 2-D cross product magnitude (= area of parallelogram)
        cross = abs(abx * bcy - aby * bcx)
        if cross < tol:
            return True
    return False


def compute_homography(
    src_pts: List[Tuple[int, int]],
    warped_w: int,
    warped_h: int,
) -> np.ndarray:
    """Compute the 3×3 perspective-transform matrix mapping *src_pts* to the
    four corners of a ``warped_w × warped_h`` output frame.

    Destination corners (in order matching *src_pts*):
        (0, 0), (warped_w, 0), (warped_w, warped_h), (0, warped_h)

    Uses ``cv2.getPerspectiveTransform`` internally.

    Requirements: 1.3
    """
    if len(src_pts) != 4:
        raise ValueError(f"compute_homography expects exactly 4 source points, got {len(src_pts)}")

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(
        [(0, 0), (warped_w, 0), (warped_w, warped_h), (0, warped_h)],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def collect_calibration_points(frame: np.ndarray) -> List[Tuple[int, int]]:
    """Open a window and collect exactly 4 left-click points from the user.

    After each click the frame is redrawn with all collected points shown as
    filled circles labelled "1", "2", "3", "4".  A status message in the
    top-left corner tells the user which point to click next.

    Returns a list of 4 (x, y) tuples in click order.

    Requirements: 1.1, 1.2
    """
    _CAL_WIN = "Calibration — click 4 ground-plane reference points"
    _POINT_COLOUR  = (0, 255, 255)   # yellow
    _LABEL_COLOUR  = (255, 255, 255) # white
    _STATUS_COLOUR = (0, 200, 255)   # orange-ish

    collected: List[Tuple[int, int]] = []

    def _redraw() -> None:
        """Redraw the calibration window with current points and status."""
        display = frame.copy()
        n = len(collected)

        # Status message
        if n < 4:
            msg = f"Click point {n + 1}/4"
        else:
            msg = "4 points collected — press y to confirm, r to redo"
        cv2.putText(display, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, _STATUS_COLOUR, 2, cv2.LINE_AA)

        # Draw collected points
        for i, (px, py) in enumerate(collected):
            cv2.circle(display, (px, py), 8, _POINT_COLOUR, -1)
            cv2.circle(display, (px, py), 9, (0, 0, 0), 1)   # thin black border
            cv2.putText(display, str(i + 1), (px + 12, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _LABEL_COLOUR, 2, cv2.LINE_AA)

        cv2.imshow(_CAL_WIN, display)

    def _mouse_cb(event, x, y, flags, param) -> None:  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN and len(collected) < 4:
            collected.append((x, y))
            print(f"  [calib] Point {len(collected)}: ({x}, {y})")
            _redraw()

    cv2.namedWindow(_CAL_WIN)
    cv2.setMouseCallback(_CAL_WIN, _mouse_cb)
    _redraw()

    print("\n[calib] Click exactly 4 ground-plane reference points on the frame.")
    print("        Press 'q' to abort without saving.\n")

    while len(collected) < 4:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(_CAL_WIN)
            print("[calib] Aborted — fewer than 4 points collected.")
            return []

    # All 4 points collected — keep window open so caller can show confirm prompt
    _redraw()
    cv2.destroyWindow(_CAL_WIN)
    return list(collected)


def show_warp_preview(
    frame: np.ndarray,
    H: np.ndarray,
    warped_size: Tuple[int, int],
) -> int:
    """Apply *H* to *frame* and display the result in a preview window.

    Overlays a short instruction string on the warped image so the user knows
    which keys to press.  Blocks until the user presses a key, then returns
    the key code so the caller can implement y / r / q logic.

    Parameters
    ----------
    frame:       Original (un-warped) video frame.
    H:           3×3 homography matrix from ``compute_homography``.
    warped_size: ``(width, height)`` of the output warped frame.

    Returns
    -------
    int
        The key code of the key pressed by the user (``cv2.waitKey`` value).

    Requirements: 1.4
    """
    _PREVIEW_WIN = "Warp Preview \u2014 press y to accept, r to redo"

    warped = cv2.warpPerspective(frame, H, warped_size)

    # Overlay instructions
    instructions = "Press y to accept, r to redo, q to quit"
    cv2.putText(
        warped, instructions,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 255), 2, cv2.LINE_AA,
    )

    cv2.imshow(_PREVIEW_WIN, warped)

    # Block until a key is pressed
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key != 255:  # 255 means no key yet
            return key


def interactive_crop(frame: np.ndarray) -> Optional[List[int]]:
    """Let the user drag a rectangle to select a crop region.

    Shows the frame in a window.  The user drags left-click to draw a
    rectangle.  Press Enter to confirm or Esc to cancel (no crop).

    Returns ``[x, y, w, h]`` on confirm, or ``None`` on cancel.
    """
    _CROP_WIN = "Crop — drag to select region, Enter=confirm, Esc=no crop"
    crop_rect: List[Optional[List[int]]] = [None]   # mutable container
    drag: List[Optional[tuple]] = [None]

    def _draw(disp: np.ndarray) -> np.ndarray:
        if crop_rect[0]:
            x, y, w, h = crop_rect[0]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(disp, f"Crop: {w}x{h} at ({x},{y})",
                        (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(disp, "Drag to select crop region | Enter=confirm | Esc=skip",
                        (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        return disp

    def _mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag[0] = (x, y)
            crop_rect[0] = None
        elif event == cv2.EVENT_MOUSEMOVE and drag[0]:
            x0, y0 = drag[0]
            disp = _draw(frame.copy())
            cv2.rectangle(disp, (x0, y0), (x, y), (0, 255, 255), 2)
            cv2.imshow(_CROP_WIN, disp)
        elif event == cv2.EVENT_LBUTTONUP and drag[0]:
            x0, y0 = drag[0]
            xs, ys = min(x0, x), min(y0, y)
            w, h = abs(x - x0), abs(y - y0)
            if w >= 20 and h >= 20:
                crop_rect[0] = [xs, ys, w, h]
            drag[0] = None
            cv2.imshow(_CROP_WIN, _draw(frame.copy()))

    cv2.namedWindow(_CROP_WIN)
    cv2.setMouseCallback(_CROP_WIN, _mouse)
    cv2.imshow(_CROP_WIN, _draw(frame.copy()))

    print("\n[crop] Drag to select crop region.")
    print("       Enter = confirm, Esc = skip (use full frame)\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:   # Enter
            cv2.destroyWindow(_CROP_WIN)
            return crop_rect[0]
        elif key == 27:  # Esc
            cv2.destroyWindow(_CROP_WIN)
            print("[crop] Skipped — using full frame.")
            return None


def run_calibration(
    frame: np.ndarray,
    warped_w: int,
    warped_h: int,
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """Calibration state machine: collect 4 points → check collinearity →
    compute H → show preview → handle y / r / q.

    Parameters
    ----------
    frame:    The video frame on which the user will click reference points.
    warped_w: Width  of the desired warped output frame in pixels.
    warped_h: Height of the desired warped output frame in pixels.

    Returns
    -------
    ``(H, (warped_w, warped_h))`` when the user accepts the preview (``y``).
    ``None`` when the user aborts (``q``) or the preview window is closed.

    Requirements: 1.4, 1.5, 1.9
    """
    _PREVIEW_WIN = "Warp Preview \u2014 press y to accept, r to redo"

    while True:
        # ── Step 1: collect 4 calibration points ──────────────────────────
        pts = collect_calibration_points(frame)

        # Empty list means the user pressed q before 4 points → abort (Req 1.9)
        if not pts:
            return None

        # ── Step 2: collinearity check (Req 1.8) ──────────────────────────
        if is_collinear(pts):
            print(
                "[calib] ERROR: The selected points are collinear or nearly "
                "collinear.\n"
                "        Please select 4 points that form a proper quadrilateral."
            )
            continue  # retry point selection

        # ── Step 3: compute homography (Req 1.3) ──────────────────────────
        try:
            H = compute_homography(pts, warped_w, warped_h)
        except Exception as exc:  # noqa: BLE001
            print(f"[calib] ERROR computing homography: {exc}. Retrying.")
            continue

        # ── Step 4: show warped preview and wait for user decision (Req 1.4) ─
        key = show_warp_preview(frame, H, (warped_w, warped_h))

        # Close the preview window regardless of which key was pressed
        cv2.destroyWindow(_PREVIEW_WIN)

        if key == ord('y'):
            # Accept → proceed to slot marking (Req 1.5)
            print("[calib] Warp preview accepted.")
            return H, (warped_w, warped_h)

        elif key == ord('r'):
            # Redo → loop back to point selection
            print("[calib] Redoing point selection.")
            continue

        else:
            # q or any other key → exit without saving (Req 1.9)
            print("[calib] Calibration aborted.")
            return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(source: Source) -> None:
    global current_frame, parking_slots, window_name
    global saved_width, saved_height, size_saved

    # Load existing slots for this source
    cfg = load_slots(source)
    parking_slots = [(x, y, cfg.rect_w, cfg.rect_h) for x, y in cfg.slots]
    saved_width  = cfg.rect_w  if cfg.slots else None
    saved_height = cfg.rect_h  if cfg.slots else None
    size_saved   = bool(cfg.slots)

    # ── Slot drawing mode: "rect" or "poly" ─────────────────────────────
    slot_mode: str = "rect"   # toggled by pressing 'p'

    # ── Entry point (mutable so key 'e' can update it) ───────────────────
    entry_point: List[int] = list(source.entry_point)

    # ── Obstacle polygons (loaded from cfg, editable with key 'o') ───────
    obstacle_polygons: List[List[List[int]]] = [list(p) for p in cfg.obstacles]
    # ── Polygon slots (loaded from cfg, editable with key 'p') ───────────
    polygon_slots: List[List[List[int]]] = [list(p) for p in cfg.poly_slots]
    # Current polygon being drawn (list of [x,y] points)
    current_obstacle: List[List[int]] = []
    current_poly_slot: List[List[int]] = []   # accumulates clicks in poly mode
    obstacle_mode: bool = False   # True while drawing an obstacle polygon
    entry_mode: bool = False      # True while waiting for a single click to set entry

    cap = cv2.VideoCapture(source.uri)
    if not cap.isOpened():
        # Try integer index for webcam sources like "0"
        try:
            cap = cv2.VideoCapture(int(source.uri))
        except ValueError:
            pass
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source.uri}")
        return

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        cap.release()
        return

    # Keep the original (un-warped) frame so recalibration can re-run on it
    original_frame = frame.copy()

    # ── Crop region ───────────────────────────────────────────────────────
    # Load existing crop; if none, offer crop tool on first run (new source)
    crop_region: Optional[List[int]] = cfg.crop_region
    is_new_source = (not cfg.slots and not cfg.poly_slots
                     and cfg.homography_matrix is None
                     and cfg.crop_region is None)

    if is_new_source:
        print("\n" + "="*60)
        print("NEW SOURCE SETUP")
        print("="*60)
        print("Choose how to prepare the frame before marking slots:")
        print("  1. Crop only     — drag to select a region, no perspective fix")
        print("  2. Calibrate     — click 4 ground points for perspective correction")
        print("  3. Crop + Calibrate — crop first, then calibrate on the cropped view")
        print("  4. Skip          — use the full frame as-is")
        print()
        choice = input("Choice [1/2/3/4, default=4]: ").strip() or "4"
        print()

        if choice in ("1", "3"):
            print("[setup] Drag to select the crop region on the frame.")
            crop_region = interactive_crop(original_frame)
            if crop_region:
                print(f"[setup] Crop set: {crop_region}")
            else:
                print("[setup] No crop selected.")

    # Apply crop to get the working frame
    if crop_region:
        cx, cy, cw, ch = crop_region
        working_frame = original_frame[cy:cy+ch, cx:cx+cw].copy()
    else:
        working_frame = original_frame.copy()

    # ── Homography setup ──────────────────────────────────────────────────
    H: Optional[np.ndarray] = None
    warped_size: Optional[Tuple[int, int]] = None

    if cfg.homography_matrix is not None:
        # Existing matrix — load and apply on the working (possibly cropped) frame
        H = np.array(cfg.homography_matrix, dtype=np.float32)
        warped_size = tuple(cfg.warped_size)  # type: ignore[arg-type]
        current_frame = cv2.warpPerspective(working_frame, H, warped_size)
        print("[mark] Loaded existing homography — skipping calibration.")
        print(f"[mark] Warped size: {warped_size[0]}x{warped_size[1]}")
    elif is_new_source and choice in ("2", "3"):
        # Run calibration on the working (possibly cropped) frame
        orig_h, orig_w = working_frame.shape[:2]
        result = run_calibration(working_frame, orig_w, orig_h)
        if result is None:
            print("[mark] Calibration aborted — exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return
        H, warped_size = result
        current_frame = cv2.warpPerspective(working_frame, H, warped_size)
    else:
        # No homography — use working frame directly
        current_frame = working_frame.copy()

    window_name = f"Mark slots — {source.name}  [{source.id}]"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    _refresh()

    print(f"\n{'='*60}")
    print(f"Source : {source.name}  ({source.uri})")
    print(f"Slots  : {source.slots_file}")
    print(f"Loaded : {len(parking_slots)} rect + {len(polygon_slots)} poly slots")
    print(f"{'='*60}")
    print("  ── Slot drawing (current mode shown in bottom bar) ──")
    print("  p          → toggle RECT / POLYGON slot mode")
    print("  Left-click → add slot (rect: drag to set, polygon: click 4 corners)")
    print("  Right-click→ undo last slot of current type")
    print("  z          → undo last polygon slot (any mode)")
    print("  r          → clear ALL rect slots")
    print("  c          → reset rect default size (drag again)")
    print("  ── Other ───────────────────────────────────────────")
    print("  e  → set entry point   o  → draw obstacle (Enter=close, Esc=cancel)")
    print("  x  → undo last obstacle")
    print("  s  → save   h  → recalibrate homography")
    print("  n  → next frame   q → quit (auto-save)")
    print(f"{'='*60}\n")

    frame_count = 0

    # ── Overlay helper ────────────────────────────────────────────────────
    def _full_refresh() -> None:
        if current_frame is None:
            return
        disp = current_frame.copy()

        # Obstacle polygons (dark red fill)
        for poly in obstacle_polygons:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                ov = disp.copy()
                cv2.fillPoly(ov, [pts], (0, 0, 180))
                cv2.addWeighted(ov, 0.35, disp, 0.65, 0, disp)
                cv2.polylines(disp, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # In-progress obstacle polygon
        if current_obstacle:
            for pt in current_obstacle:
                cv2.circle(disp, tuple(pt), 4, (0, 140, 255), -1)
            for i in range(len(current_obstacle) - 1):
                cv2.line(disp, tuple(current_obstacle[i]),
                         tuple(current_obstacle[i+1]), (0, 140, 255), 1)

        # Saved polygon slots (cyan outline)
        n_rect = len(parking_slots)
        for j, poly in enumerate(polygon_slots):
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(disp, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
                cx = int(np.mean([p[0] for p in poly]))
                cy = int(np.mean([p[1] for p in poly]))
                cv2.putText(disp, str(n_rect + j + 1), (cx - 6, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # In-progress polygon slot preview
        if current_poly_slot:
            for pt in current_poly_slot:
                cv2.circle(disp, tuple(pt), 6, (0, 200, 255), -1)
            for i in range(len(current_poly_slot) - 1):
                cv2.line(disp, tuple(current_poly_slot[i]),
                         tuple(current_poly_slot[i+1]), (0, 200, 255), 2)
            if len(current_poly_slot) >= 3:
                cv2.line(disp, tuple(current_poly_slot[-1]),
                         tuple(current_poly_slot[0]), (0, 200, 255), 1)
            # Show which point comes next
            cv2.putText(disp, f"Poly point {len(current_poly_slot)+1}/4",
                        (current_poly_slot[-1][0] + 8, current_poly_slot[-1][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Entry point
        ex, ey = entry_point
        cv2.circle(disp, (ex, ey), 10, (255, 0, 255), -1)
        cv2.putText(disp, "ENTRY", (ex - 25, ey - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # ── Bottom status bar ─────────────────────────────────────────────
        bar_y = disp.shape[0] - 10
        if entry_mode:
            cv2.putText(disp, "[ ENTRY MODE ] click to place entry point",
                        (10, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        elif obstacle_mode:
            cv2.putText(disp,
                        f"[ OBSTACLE ] {len(current_obstacle)} pts — Enter=save  Esc=cancel",
                        (10, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
        elif slot_mode == "poly":
            n_in = len(current_poly_slot)
            msg = (f"[ POLYGON SLOT ] click point {n_in+1}/4"
                   if n_in < 4 else "[ POLYGON SLOT ] 4 pts — saving…")
            cv2.putText(disp, msg, (10, bar_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        else:
            rect_info = (f"size {saved_width}x{saved_height}" if size_saved and saved_width
                         else "drag to set size")
            cv2.putText(disp, f"[ RECTANGLE SLOT ] {rect_info}  — press P for polygon",
                        (10, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 160), 2)

        draw_rectangles(disp, parking_slots)
        cv2.imshow(window_name, disp)

    # Override mouse callback to route clicks based on slot_mode
    def _extended_mouse_cb(event, x, y, flags, param) -> None:
        nonlocal entry_mode, obstacle_mode, slot_mode

        # Entry mode: one click places the entry point, then exits
        if entry_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                entry_point[0], entry_point[1] = x, y
                print(f"  [entry] Entry point set to ({x}, {y})")
                entry_mode = False
                _full_refresh()
            return

        # Obstacle mode: clicks add polygon vertices
        if obstacle_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                current_obstacle.append([x, y])
                print(f"  [obstacle] Point {len(current_obstacle)}: ({x}, {y})")
                _full_refresh()
            return

        # Polygon slot mode: accumulate 4 clicks, auto-save on 4th
        if slot_mode == "poly":
            if event == cv2.EVENT_LBUTTONDOWN:
                current_poly_slot.append([x, y])
                print(f"  [poly] Point {len(current_poly_slot)}/4: ({x}, {y})")
                _full_refresh()
                if len(current_poly_slot) == 4:
                    polygon_slots.append([pt[:] for pt in current_poly_slot])
                    slot_num = len(parking_slots) + len(polygon_slots)
                    print(f"  [poly] Saved polygon slot #{slot_num}")
                    current_poly_slot.clear()
                    _full_refresh()
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right-click: undo last completed polygon slot OR last in-progress point
                if current_poly_slot:
                    removed_pt = current_poly_slot.pop()
                    print(f"  [poly] Removed last point, {len(current_poly_slot)} remaining")
                    _full_refresh()
                elif polygon_slots:
                    polygon_slots.pop()
                    print(f"  [poly] Removed last polygon slot. Remaining: {len(polygon_slots)}")
                    _full_refresh()
            return

        # Rectangle slot mode: normal drag/click behaviour
        mouse_callback(event, x, y, flags, param)
        _full_refresh()

    cv2.setMouseCallback(window_name, _extended_mouse_cb)
    _full_refresh()

    def _build_cfg() -> SlotConfig:
        w = saved_width  or cfg.rect_w
        h = saved_height or cfg.rect_h
        if H is not None:
            hm = H.tolist()
            ws: Optional[List[int]] = list(warped_size)  # type: ignore[arg-type]
        else:
            hm = None
            ws = None
        return SlotConfig(
            rect_w=w, rect_h=h,
            threshold=cfg.threshold,
            slots=[(x, y) for x, y, *_ in parking_slots],
            homography_matrix=hm,
            warped_size=ws,
            obstacles=obstacle_polygons,
            poly_slots=polygon_slots,
            crop_region=crop_region,
        )

    def _save_entry_point() -> None:
        sources = load_sources()
        for s in sources:
            if s.id == source.id:
                s.entry_point = tuple(entry_point)  # type: ignore[assignment]
                break
        save_sources(sources)
        print(f"  [entry] Saved entry point {tuple(entry_point)} to sources.json")

    while True:
        key = cv2.waitKey(20) & 0xFF

        # ── Obstacle mode keys (Enter=save, Esc=cancel) ───────────────────
        if obstacle_mode:
            if key == 13:  # Enter — close polygon
                if len(current_obstacle) >= 3:
                    obstacle_polygons.append([pt[:] for pt in current_obstacle])
                    print(f"  [obstacle] Polygon saved ({len(current_obstacle)} points). "
                          f"Total: {len(obstacle_polygons)}")
                else:
                    print("  [obstacle] Need at least 3 points — cancelled.")
                current_obstacle.clear()
                obstacle_mode = False
                _full_refresh()
            elif key == 27:  # Esc — cancel
                current_obstacle.clear()
                obstacle_mode = False
                print("  [obstacle] Cancelled.")
                _full_refresh()
            continue   # swallow all other keys while in obstacle mode

        # ── Normal / slot-mode keys ───────────────────────────────────────
        if key == ord('q'):
            if parking_slots or polygon_slots:
                save_slots(source, _build_cfg())
                _save_entry_point()
                print(f"[mark] Auto-saved {len(parking_slots)} rect + "
                      f"{len(polygon_slots)} poly slots on exit.")
            break

        elif key == ord('s'):
            save_slots(source, _build_cfg())
            _save_entry_point()

        elif key == ord('p'):
            # Toggle between rect and polygon slot mode
            if slot_mode == "rect":
                slot_mode = "poly"
                current_poly_slot.clear()
                print("[mark] Switched to POLYGON SLOT mode — click 4 corners per slot. "
                      "Press P again to return to rectangle mode.")
            else:
                slot_mode = "rect"
                current_poly_slot.clear()
                print("[mark] Switched to RECTANGLE SLOT mode.")
            _full_refresh()

        elif key == ord('z'):
            # Undo last polygon slot (works in any slot mode)
            if current_poly_slot:
                current_poly_slot.pop()
                print(f"  [poly] Removed last in-progress point.")
                _full_refresh()
            elif polygon_slots:
                polygon_slots.pop()
                print(f"  [poly] Removed last polygon slot. Remaining: {len(polygon_slots)}")
                _full_refresh()
            else:
                print("  [poly] No polygon slots to remove.")

        elif key == ord('e'):
            entry_mode = not entry_mode
            if entry_mode:
                print("[mark] ENTRY MODE — click anywhere to place the entry point.")
            else:
                print("[mark] Entry mode cancelled.")
            _full_refresh()

        elif key == ord('o'):
            obstacle_mode = True
            current_obstacle.clear()
            print("[mark] OBSTACLE MODE — click to add polygon points. "
                  "Enter to close, Esc to cancel.")
            _full_refresh()

        elif key == ord('x'):
            if obstacle_polygons:
                removed = obstacle_polygons.pop()
                print(f"  [obstacle] Removed last polygon ({len(removed)} points). "
                      f"Remaining: {len(obstacle_polygons)}")
                _full_refresh()
            else:
                print("  [obstacle] No polygons to remove.")

        elif key == ord('r'):
            parking_slots.clear()
            size_saved = False
            saved_width = saved_height = None
            print("[mark] Cleared all rect slots.")
            _full_refresh()

        elif key == ord('c'):
            size_saved = False
            saved_width = saved_height = None
            print("[mark] Size reset — drag to set new size.")
            _full_refresh()

        elif key == ord('h'):
            print("[mark] Recalibration requested — clearing existing homography.")
            orig_h, orig_w = working_frame.shape[:2]
            result = run_calibration(working_frame, orig_w, orig_h)
            if result is not None:
                H, warped_size = result
                current_frame = cv2.warpPerspective(working_frame, H, warped_size)
                print(f"[mark] Recalibration complete. New warped size: {warped_size[0]}x{warped_size[1]}")
                _full_refresh()
            else:
                print("[mark] Recalibration aborted — keeping previous homography.")

        elif key == ord('n'):
            ret, new_frame = cap.read()
            if ret:
                original_frame = new_frame.copy()
                if crop_region:
                    cx, cy, cw, ch = crop_region
                    working_frame = original_frame[cy:cy+ch, cx:cx+cw].copy()
                else:
                    working_frame = original_frame.copy()
                if H is not None and warped_size is not None:
                    current_frame = cv2.warpPerspective(working_frame, H, warped_size)
                else:
                    current_frame = working_frame.copy()
                frame_count += 1
                print(f"[mark] Frame {frame_count}")
                _full_refresh()
            else:
                print("[mark] End of video.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--list" in args:
        list_sources()
        sys.exit(0)

    # ── Add a new source interactively ───────────────────────────────────
    if "--add" in args:
        from config import add_source_interactive
        add_source_interactive()
        print("\nNow run:  python mark_parking_slots.py --source <ID>")
        sys.exit(0)

    source_id = None
    if "--source" in args:
        idx = args.index("--source")
        if idx + 1 < len(args):
            source_id = args[idx + 1]

    sources = load_sources()

    if source_id:
        source = get_source_by_id(source_id)
        if source is None:
            print(f"[ERROR] Source '{source_id}' not found in sources.json.")
            print()
            print("Options:")
            print("  1. Add it now interactively")
            print("  2. Exit and add manually via: python config.py --add")
            choice = input("Add now? [y/N]: ").strip().lower()
            if choice == 'y':
                from config import add_source_interactive
                add_source_interactive()
                source = get_source_by_id(source_id)
                if source is None:
                    print(f"[ERROR] Source '{source_id}' still not found — "
                          "make sure the ID you entered matches.")
                    sys.exit(1)
            else:
                list_sources()
                sys.exit(1)
    elif not sources:
        print("[mark] No sources defined yet.")
        print("       Adding a new source now...\n")
        from config import add_source_interactive
        add_source_interactive()
        sources = load_sources()
        if not sources:
            print("[ERROR] No sources after add — exiting.")
            sys.exit(1)
        source = sources[-1]
        print(f"[mark] Using newly added source: {source.id}")
    elif len(sources) == 1:
        source = sources[0]
        print(f"[mark] Using only source: {source.id}")
    else:
        print("Multiple sources available:")
        list_sources()
        print()
        chosen = input("Enter source ID to mark (or 'new' to add a source): ").strip()
        if chosen.lower() == 'new':
            from config import add_source_interactive
            add_source_interactive()
            sources = load_sources()
            chosen = input("Enter the ID of the source you just added: ").strip()
        source = get_source_by_id(chosen)
        if source is None:
            print(f"[ERROR] Source '{chosen}' not found.")
            sys.exit(1)

    main(source)

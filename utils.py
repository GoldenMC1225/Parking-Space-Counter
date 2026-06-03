"""
utils.py
========
Shared utilities for the parking space counter system.

Provides A* pathfinding, nearest-slot search, obstacle grid construction,
and the unified slot-rendering helper used by both the classical CV pipeline
and the YOLO pipeline.

Exposed names
-------------
    GRID_CELL
    find_nearest_free_slot
    build_obstacle_grid
    astar
    draw_slots
"""

from __future__ import annotations

import heapq
import cv2
import numpy as np
from typing import List, Optional, Tuple

from config import SlotConfig

# A* grid resolution (pixels per cell)
GRID_CELL: int = 20


# ─────────────────────────────────────────────
# A* pathfinding
# ─────────────────────────────────────────────

def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def build_obstacle_grid(
    frame_shape: Tuple[int, ...],
    cfg: SlotConfig,
    statuses: List[bool],
) -> np.ndarray:
    """Grid where occupied slots and user-defined obstacles are 1, free space is 0.

    statuses covers rect slots first, then poly slots, in that order.
    """
    h, w = frame_shape[:2]
    rows = h // GRID_CELL + 1
    cols = w // GRID_CELL + 1
    grid = np.zeros((rows, cols), dtype=np.uint8)

    n_rect = len(cfg.slots)

    # Mark occupied rectangle slots as obstacles
    for i, ((x, y), is_free) in enumerate(zip(cfg.slots, statuses[:n_rect])):
        if not is_free:
            gx1 = max(0, x // GRID_CELL)
            gy1 = max(0, y // GRID_CELL)
            gx2 = min(cols - 1, (x + cfg.rect_w) // GRID_CELL)
            gy2 = min(rows - 1, (y + cfg.rect_h) // GRID_CELL)
            grid[gy1 : gy2 + 1, gx1 : gx2 + 1] = 1

    # Mark occupied polygon slots as obstacles (rasterise each polygon)
    poly_statuses = statuses[n_rect:]
    if cfg.poly_slots:
        pixel_mask = np.zeros((h, w), dtype=np.uint8)
        for poly, is_free in zip(cfg.poly_slots, poly_statuses):
            if not is_free and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(pixel_mask, [pts], 1)
        for gr in range(rows):
            for gc in range(cols):
                py1 = gr * GRID_CELL; py2 = min(h, py1 + GRID_CELL)
                px1 = gc * GRID_CELL; px2 = min(w, px1 + GRID_CELL)
                if py2 > py1 and px2 > px1 and pixel_mask[py1:py2, px1:px2].any():
                    grid[gr, gc] = 1

    # Mark user-defined obstacle polygons (walls, pavements, no-go zones)
    if cfg.obstacles:
        pixel_mask2 = np.zeros((h, w), dtype=np.uint8)
        for poly in cfg.obstacles:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(pixel_mask2, [pts], 1)
        for gr in range(rows):
            for gc in range(cols):
                py1 = gr * GRID_CELL; py2 = min(h, py1 + GRID_CELL)
                px1 = gc * GRID_CELL; px2 = min(w, px1 + GRID_CELL)
                if py2 > py1 and px2 > px1 and pixel_mask2[py1:py2, px1:px2].any():
                    grid[gr, gc] = 1

    return grid


def astar(
    grid: np.ndarray,
    start_px: Tuple[int, int],
    goal_px: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    A* on an 8-directional grid.
    Inputs/outputs are pixel coordinates (x, y).
    Returns the path as pixel waypoints, or None if unreachable.
    """
    rows, cols = grid.shape

    def to_grid(px: Tuple[int, int]) -> Tuple[int, int]:
        return (
            max(0, min(rows - 1, px[1] // GRID_CELL)),
            max(0, min(cols - 1, px[0] // GRID_CELL)),
        )

    def to_pixel(gc: Tuple[int, int]) -> Tuple[int, int]:
        return (
            gc[1] * GRID_CELL + GRID_CELL // 2,
            gc[0] * GRID_CELL + GRID_CELL // 2,
        )

    start = to_grid(start_px)
    goal  = to_grid(goal_px)

    # If goal cell is an obstacle, find nearest free neighbour
    if grid[goal[0], goal[1]] == 1:
        found = False
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = goal[0] + dr, goal[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    goal = (nr, nc)
                    found = True
                    break
            if found:
                break

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: dict = {}
    g_score: dict   = {start: 0.0}

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path: List[Tuple[int, int]] = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return [to_pixel(n) for n in path]

        for dr, dc in directions:
            nb = (current[0] + dr, current[1] + dc)
            nr, nc = nb
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            cost = 1.414 if dr and dc else 1.0
            tg = g_score[current] + cost
            if tg < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb]   = tg
                f = tg + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f, nb))

    return None


def find_nearest_free_slot(
    entry: Tuple[int, int],
    cfg: SlotConfig,
    statuses: List[bool],
) -> Optional[int]:
    """Return index of the free slot closest to entry (Euclidean).

    statuses covers rect slots first, then poly slots.
    Returned index is into the combined list (rect slots first).
    """
    best_idx: Optional[int] = None
    best_dist = float("inf")

    # Rectangle slots
    for i, ((x, y), is_free) in enumerate(zip(cfg.slots, statuses)):
        if is_free:
            cx = x + cfg.rect_w // 2
            cy = y + cfg.rect_h // 2
            d = ((cx - entry[0]) ** 2 + (cy - entry[1]) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = i

    # Polygon slots (offset index by n_rect)
    n_rect = len(cfg.slots)
    poly_statuses = statuses[n_rect:]
    for j, (poly, is_free) in enumerate(zip(cfg.poly_slots, poly_statuses)):
        if is_free and len(poly) >= 3:
            pts = np.array(poly, dtype=np.float32)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            d = ((cx - entry[0]) ** 2 + (cy - entry[1]) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = n_rect + j

    return best_idx


# ─────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────

def draw_slots(
    frame: np.ndarray,
    cfg: SlotConfig,
    statuses: List[bool],
    free_count: int,
    nearest_idx: Optional[int],
    path: Optional[List[Tuple[int, int]]],
    entry: Tuple[int, int],
    source_name: str,
    extra_overlay: Optional[str] = None,   # e.g. "CACHED" or "LIVE"
) -> np.ndarray:
    """
    Draw parking slot rectangles, A* path, entry point, and HUD onto *frame*.

    Parameters
    ----------
    frame        : BGR image to annotate (modified in-place and returned).
    cfg          : Slot configuration (positions, dimensions).
    statuses     : Per-slot occupancy flags (True = free).
    free_count   : Pre-computed count of free slots (used in HUD).
    nearest_idx  : Index of the nearest free slot, or None.
    path         : A* path as a list of (x, y) pixel waypoints, or None.
    entry        : Entry-point pixel coordinate (x, y).
    source_name  : Display name shown in the HUD.
    extra_overlay: Optional text rendered in the top-right corner.
                   "CACHED" is drawn in yellow; "LIVE" is drawn in green;
                   any other non-None string is drawn in white.
    """
    fh, fw = frame.shape[:2]
    n_rect = len(cfg.slots)

    # Draw user-defined obstacle polygons (semi-transparent red fill + border)
    if cfg.obstacles:
        overlay = frame.copy()
        for poly in cfg.obstacles:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], (0, 0, 180))   # dark red fill
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        for poly in cfg.obstacles:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw rectangle slots
    for i, ((x, y), is_free) in enumerate(zip(cfg.slots, statuses[:n_rect])):
        x1 = x + 10
        x2 = x + cfg.rect_w - 11
        y1 = y + 4
        y2 = y + cfg.rect_h
        if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
            continue

        if i == nearest_idx:
            color, thick = (0, 255, 255), 4   # yellow — nearest free
        elif is_free:
            color, thick = (0, 255, 0), 3     # green — free
        else:
            color, thick = (0, 0, 255), 2     # red — occupied

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, str(i + 1), (x1 + 2, y1 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Draw polygon slots
    poly_statuses = statuses[n_rect:]
    for j, (poly, is_free) in enumerate(zip(cfg.poly_slots, poly_statuses)):
        if len(poly) < 3:
            continue
        global_idx = n_rect + j
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)

        if global_idx == nearest_idx:
            color, thick = (0, 255, 255), 4
        elif is_free:
            color, thick = (0, 255, 0), 3
        else:
            color, thick = (0, 0, 255), 2

        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thick)
        # Label at centroid
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        cv2.putText(frame, str(global_idx + 1), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # A* path
    if path and len(path) > 1:
        for j in range(len(path) - 1):
            cv2.line(frame, path[j], path[j + 1], (255, 165, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, path[0],  8, (0, 255, 255), -1)
        cv2.circle(frame, path[-1], 8, (0, 255, 0),   -1)

    # Entry point marker
    cv2.circle(frame, entry, 10, (255, 0, 255), -1)
    cv2.putText(frame, "ENTRY", (entry[0] - 25, entry[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # HUD — total = rect + poly slots
    total_slots = len(cfg.slots) + len(cfg.poly_slots)
    cv2.rectangle(frame, (0, 0), (360, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"{source_name}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, f"Free: {free_count} / {total_slots}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
    if nearest_idx is not None:
        cv2.putText(frame, f"Nearest free: Slot #{nearest_idx + 1}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # YOLO CACHED / LIVE indicator (top-right corner)
    if extra_overlay is not None:
        if extra_overlay == "CACHED":
            overlay_color = (0, 255, 255)   # yellow
        elif extra_overlay == "LIVE":
            overlay_color = (0, 255, 0)     # green
        else:
            overlay_color = (255, 255, 255) # white

        (text_w, text_h), _ = cv2.getTextSize(
            extra_overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        tx = fw - text_w - 10
        ty = text_h + 10
        cv2.putText(frame, extra_overlay, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)

    return frame


__all__ = [
    "GRID_CELL",
    "find_nearest_free_slot",
    "build_obstacle_grid",
    "astar",
    "draw_slots",
    "apply_crop",
]


def apply_crop(frame: np.ndarray, cfg: SlotConfig) -> np.ndarray:
    """Crop *frame* to ``cfg.crop_region`` if one is defined.

    ``crop_region`` is stored as ``[x, y, width, height]`` in original-frame
    pixel coordinates.  All slot coordinates are relative to the cropped frame,
    so every pipeline must call this before any detection or rendering.

    Returns the original frame unchanged when ``crop_region`` is None.
    """
    if cfg.crop_region is None:
        return frame
    x, y, w, h = cfg.crop_region
    fh, fw = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]

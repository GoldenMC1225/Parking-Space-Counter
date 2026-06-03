"""
detect_improved.py
==================
Parking space counter with A* pathfinding.
All source/slot configuration is loaded from sources.json — no hardcoding.

Usage
-----
  python detect_improved.py                        # picks first source
  python detect_improved.py --source carpark_main
  python detect_improved.py --list                 # list available sources
"""

from __future__ import annotations

import sys
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional

from config import (
    load_sources, get_source_by_id, load_slots, list_sources,
    Source, SlotConfig,
)
from utils import (
    find_nearest_free_slot,
    build_obstacle_grid,
    astar,
    draw_slots,
    apply_crop,
    GRID_CELL,
)


# ─────────────────────────────────────────────
# Image processing
# ─────────────────────────────────────────────

def convert_grayscale(frame: np.ndarray) -> np.ndarray:
    """Return a contour-only image (white edges on black)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(frame)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), thickness=2)
    return canvas


def check_slots(
    grayscale_frame: np.ndarray,
    cfg: SlotConfig,
) -> Tuple[List[bool], int]:
    """Check each slot against the contour image.

    Handles both rectangle slots (bounding-box crop + pixel count) and
    polygon slots (mask crop + pixel count inside polygon).
    Returns (statuses, free_count) where True = free.
    Combined order: rect slots first, then poly slots.
    """
    fh, fw = grayscale_frame.shape[:2]
    statuses: List[bool] = []
    free_count = 0

    # ── Rectangle slots ───────────────────────────────────────────────────
    for x, y in cfg.slots:
        x1 = x + 10
        x2 = x + cfg.rect_w - 11
        y1 = y + 4
        y2 = y + cfg.rect_h

        if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh or x2 <= x1 or y2 <= y1:
            statuses.append(False)
            continue

        crop = grayscale_frame[y1:y2, x1:x2]
        if crop.size == 0:
            statuses.append(False)
            continue

        gray_crop = (
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if len(crop.shape) == 3
            else crop
        )
        count = cv2.countNonZero(gray_crop)
        is_free = count < cfg.threshold
        statuses.append(is_free)
        if is_free:
            free_count += 1

    # ── Polygon slots ─────────────────────────────────────────────────────
    for poly in cfg.poly_slots:
        if len(poly) < 3:
            statuses.append(False)
            continue

        pts = np.array(poly, dtype=np.int32)
        # Bounding box of the polygon
        bx1, by1 = pts[:, 0].min(), pts[:, 1].min()
        bx2, by2 = pts[:, 0].max(), pts[:, 1].max()
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(fw, bx2), min(fh, by2)

        if bx2 <= bx1 or by2 <= by1:
            statuses.append(False)
            continue

        # Create a mask for the polygon region
        mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
        mask_crop = mask[by1:by2, bx1:bx2]

        # Get grayscale crop
        region = grayscale_frame[by1:by2, bx1:bx2]
        gray_region = (
            cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            if len(region.shape) == 3
            else region
        )

        # Count non-zero pixels inside the polygon only
        masked = cv2.bitwise_and(gray_region, gray_region, mask=mask_crop)
        count = cv2.countNonZero(masked)
        is_free = count < cfg.threshold
        statuses.append(is_free)
        if is_free:
            free_count += 1

    return statuses, free_count



# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def run(source: Source) -> None:
    cfg = load_slots(source)
    if not cfg.slots and not cfg.poly_slots:
        print(f"[WARN] No slots defined for '{source.id}'.")
        print(f"       Run: python mark_parking_slots.py --source {source.id}")

    # ── Resolve homography ───────────────────────────────────────────────
    H_cl: Optional[np.ndarray] = None
    warped_size_cl: Optional[Tuple[int, int]] = None
    if cfg.homography_matrix is not None:
        H_cl = np.array(cfg.homography_matrix, dtype=np.float64)
        if cfg.warped_size is not None:
            warped_size_cl = (int(cfg.warped_size[0]), int(cfg.warped_size[1]))

    cap = cv2.VideoCapture(source.uri)
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(source.uri))
        except ValueError:
            pass
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source.uri}")
        return

    entry = tuple(source.entry_point)

    # A* cache — recompute only when slot statuses change
    cached_statuses: Optional[List[bool]] = None
    cached_path:     Optional[List[Tuple[int, int]]] = None
    cached_nearest:  Optional[int] = None

    t_prev = time.time()
    is_file = not (source.uri.startswith("rtsp") or source.uri.isdigit())

    print(f"[detect] Running '{source.name}'  —  press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[detect] Stream ended.")
                break

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
            working, cfg, statuses, free_count,
            cached_nearest, cached_path, entry, source.name,
        )

        t_now = time.time()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        cv2.putText(out, f"FPS: {fps:.1f}", (out.shape[1] - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow(f"Parking Counter — {source.name}", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--list" in args:
        list_sources()
        sys.exit(0)

    source_id = None
    if "--source" in args:
        idx = args.index("--source")
        if idx + 1 < len(args):
            source_id = args[idx + 1]

    sources = load_sources()
    if not sources:
        print("[ERROR] No sources defined in sources.json")
        print("        Run: python config.py --add")
        sys.exit(1)

    if source_id:
        source = get_source_by_id(source_id)
        if source is None:
            print(f"[ERROR] Source '{source_id}' not found.")
            list_sources()
            sys.exit(1)
    elif len(sources) == 1:
        source = sources[0]
    else:
        print("Multiple sources available:")
        list_sources()
        print()
        chosen = input("Enter source ID to run: ").strip()
        source = get_source_by_id(chosen)
        if source is None:
            print(f"[ERROR] Source '{chosen}' not found.")
            sys.exit(1)

    run(source)

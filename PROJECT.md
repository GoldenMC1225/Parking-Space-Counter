# Parking Space Counter

A real-time parking lot occupancy detection system that supports multiple video sources, two detection pipelines, perspective correction, crop-based ROI, A* guided navigation, and a web-based monitoring interface.

---

## What It Does

The system monitors parking lots through CCTV cameras and answers two questions in real time:

1. **Which slots are occupied, which are free?**
2. **What is the shortest driveable path from the entrance to the nearest free slot?**

It supports multiple cameras simultaneously, lets the user switch between a classical CV pipeline (CPU-only) and a YOLO deep learning pipeline (GPU), and serves a live annotated video feed through a web browser.

---

## How It Works

### Frame Processing Pipeline

```
Video frame (file / RTSP / webcam)
    │
    ▼
apply_crop()            ← optional ROI crop [x, y, w, h] stored in slots JSON
    │
    ▼
warpPerspective(H)      ← optional homography correction (bird's-eye view)
    │                      Both pipelines apply crop + warp before detection.
    │
    ├─── Classical CV path ──────────────────────────────────────────────
    │    convert_grayscale() → check_slots() → statuses (True = free)
    │
    └─── YOLO path ──────────────────────────────────────────────────────
         detect on cropped original frame (NOT warped) → filter_detections()
             → project_boxes_to_warped(H)   ← projects both bbox-center
             →                                  and bottom-center through H
             → assign_slots()               ← tests center OR bottom-center
             → statuses (True = occupied)       against slot boundaries
    │
    ▼
find_nearest_free_slot()   ← Euclidean search over free slots (rect + poly)
    │
    ▼
build_obstacle_grid()      ← occupied slots + user obstacle polygons → grid
    │
    ▼
astar(grid, entry, goal)   ← 8-directional A* pathfinding
    │
    ▼
draw_slots()               ← annotate warped frame with colours, path, HUD
    │
    ▼
Display (OpenCV window) or serve via MJPEG stream (FastAPI)
```

### Detection Pipelines

| Pipeline | File | Method |
|---|---|---|
| Classical CV | `detect_improved.py` | Grayscale threshold + contour pixel count |
| YOLO | `detect_yolo.py` | YOLOv8/RT-DETR/D-FINE + point projection + slot assignment |

Classical CV is fast, runs on CPU, but sensitive to lighting and shadows.
YOLO is robust to lighting variation and distinguishes vehicle types, but requires a GPU for real-time speed.

### YOLO Detection Fix — Detect on Original Frame

YOLO was trained on side-view COCO images. Running it on a top-down warped frame causes low confidence and wrong class labels. The pipeline instead:

1. Runs YOLO on the **cropped original frame** (side-view — what YOLO was trained on)
2. Projects each detection's **bbox-center** and **bottom-center** through homography `H`
3. Tests both projected points against slot boundaries in warped space

This means a slot is marked occupied if either the bbox center **or** the bottom-center lands inside it — necessary because top-down cameras often place the bbox bottom edge outside the slot polygon.

### Slot Assignment Logic

- **Rectangle slots** — bounding-box containment test on center or bottom-center
- **Polygon slots** — `cv2.pointPolygonTest` on center or bottom-center (handles angled CCTV views where slots appear as trapezoids)

---

## File Structure

```
├── config.py               Data models + JSON load/save helpers
├── utils.py                A*, obstacle grid, nearest-slot search, draw_slots, apply_crop
├── mark_parking_slots.py   Interactive slot marking tool (OpenCV GUI)
├── detect_improved.py      Classical CV detection pipeline
├── detect_yolo.py          YOLO detection pipeline
├── app.py                  FastAPI web server (MJPEG stream, mode switch)
├── evaluate.py             Precision / Recall / FPS evaluation framework
├── sources.json            List of video/RTSP sources
├── slots/
│   └── <source_id>.json    Slot coords, homography, crop region, obstacles per source
├── requirements.txt        Pinned Python dependencies
├── PROJECT.MD              This file
└── Parking_Space_Counter.ipynb  Google Colab / local Jupyter notebook
```

---

## Important Functions

### `utils.py`

| Function | Purpose |
|---|---|
| `astar(grid, start_px, goal_px)` | 8-directional A* on a 20×20 px cell grid. Returns pixel waypoints or `None` if unreachable. |
| `build_obstacle_grid(frame_shape, cfg, statuses)` | Rasterises occupied rect/poly slots and user-drawn obstacle polygons into a 2-D numpy grid. |
| `find_nearest_free_slot(entry, cfg, statuses)` | Euclidean nearest-free-slot search covering both rect and polygon slots. |
| `draw_slots(frame, cfg, statuses, ...)` | Draws slot outlines (green/red/yellow), A* path (orange), entry point (magenta), obstacle fills, LIVE/CACHED indicator, and HUD. |
| `apply_crop(frame, cfg)` | Crops the frame to `cfg.crop_region` `[x, y, w, h]` when set; returns unchanged otherwise. |

### `detect_yolo.py`

| Function | Purpose |
|---|---|
| `filter_detections(raw_results, conf_thresh)` | Keeps only detections whose class is in `{car, truck, bus, motorcycle}` and confidence ≥ threshold. |
| `project_boxes_to_warped(boxes, H)` | Projects **both** bbox-center `(x1,y1)` and bottom-center `(x2,y2)` of each detection through H. Returns `(cx,cy,bx,by,cls,conf)` tuples. |
| `assign_slots(boxes, cfg)` | Tests center OR bottom-center against every slot boundary. Returns `List[bool]` (True = occupied). Handles both rect and polygon slots. |
| `bottom_center(x1, y1, x2, y2)` | Returns `(int((x1+x2)/2), int(y2))` — bottom edge midpoint of a bbox. |
| `run(source, model_path, skip, conf)` | Full YOLO loop: crop → detect on original → project → assign → warp for display → A* → draw → debug window → show. |
| `FrameCache` | Dataclass holding `statuses`, `boxes`, `boxes_original`, `is_initialized` — reused between inference frames to avoid per-frame GPU cost. |

### `mark_parking_slots.py`

| Function | Purpose |
|---|---|
| `interactive_crop(frame)` | Drag-to-select crop ROI on the full frame. Returns `[x, y, w, h]` or `None`. |
| `run_calibration(frame, w, h)` | Full homography calibration: collect 4 points → collinearity check → compute H → preview → accept/redo. |
| `collect_calibration_points(frame)` | Mouse callback collecting exactly 4 click points for homography calibration. |
| `compute_homography(src_pts, w, h)` | Wraps `cv2.getPerspectiveTransform` to map 4 source points to the 4 canvas corners. |
| `interactive_crop(frame)` | Drag to select an ROI crop. Used in new-source setup. |
| `main(source)` | Interactive slot-marking loop with persistent RECT/POLY mode, obstacle drawing, entry point placement, new-source setup menu, and auto-save. |

### `app.py`

| Function | Purpose |
|---|---|
| `_classical_pipeline_loop(state)` | Background thread: crop → warp → classical CV detection → updates `state.latest_frame` and `state.status`. |
| `_yolo_pipeline_loop(state, ...)` | Background thread: crop → detect on original → project → assign → warp for display → updates shared state. |
| `switch_pipeline(state, new_mode)` | Stops current thread (3 s timeout), starts new thread. `latest_frame` preserved during transition. |
| `stream(source_id)` | FastAPI MJPEG endpoint (`multipart/x-mixed-replace`) from `state.latest_frame`. |
| `status(source_id)` | Returns JSON: `free_slots`, `total_slots`, `nearest_free_slot`, `fps`, `mode`, `source_name`. |
| `_resolve_uri(uri)` | Resolves relative file URIs against the project directory so `./video/foo.mp4` works regardless of where the server is launched from. |

### `config.py`

| Class / Function | Purpose |
|---|---|
| `SlotConfig` | Dataclass: rect slots, polygon slots, homography matrix, warped size, **crop region**, obstacle polygons, detection threshold. |
| `Source` | Dataclass: id, name, URI, slots file path, entry point. |
| `load_slots(source)` | Deserialises `SlotConfig` from JSON. Defaults to empty config if file missing. |
| `save_slots(source, cfg)` | Serialises `SlotConfig` to JSON, creating directories if needed. |
| `add_source_interactive()` | CLI prompt to add a new source. Entry point defaults to `(0,0)` — set it visually with `e` in the marker. |

---

## `SlotConfig` Fields

| Field | Type | Purpose |
|---|---|---|
| `rect_w`, `rect_h` | `int` | Default rectangle slot dimensions |
| `threshold` | `int` | Classical CV pixel-count threshold |
| `slots` | `List[Tuple[int,int]]` | Rectangle slot top-left corners |
| `poly_slots` | `List[List[List[int]]]` | Polygon slot corner points (4 pts each) |
| `homography_matrix` | `Optional[List[List[float]]]` | 3×3 perspective warp matrix |
| `warped_size` | `Optional[List[int]]` | `[width, height]` of warped canvas |
| `crop_region` | `Optional[List[int]]` | `[x, y, w, h]` ROI in original frame |
| `obstacles` | `List[List[List[int]]]` | User-drawn no-go zone polygons for A* |

---

## New Source Setup Workflow

When `mark_parking_slots.py` is run on a source with no existing configuration, it presents an interactive setup menu:

```
NEW SOURCE SETUP
════════════════
1. Crop only        — drag to select ROI, no perspective correction
2. Calibrate        — click 4 ground-plane points for bird's-eye warp
3. Crop + Calibrate — crop first, then calibrate on the cropped view
4. Skip             — use full frame as-is (default)
```

After setup, all slot coordinates are stored **relative to the cropped/warped frame**. Every pipeline applies the same crop + warp at runtime, so coordinates always match.

---

## Slot Marking Tool Controls

```
p          → toggle RECTANGLE / POLYGON slot mode (persistent)
Left-click → add slot  (rect: drag to set size / polygon: click 4 corners)
Right-click→ undo last slot of current type
z          → undo last polygon slot
r          → clear all rectangle slots
c          → reset rectangle default size (drag again)
e          → enter ENTRY MODE — click to place entry point
o          → enter OBSTACLE MODE — click vertices, Enter=save, Esc=cancel
x          → undo last obstacle polygon
h          → recalibrate homography (re-runs on working frame)
n          → advance to next frame
s          → save (slots + entry point)
q          → quit (auto-saves)
--add      → add a new source without launching the marker
```

The current slot mode and any active sub-mode are always shown in the **bottom status bar** of the window.

---

## Visual Legend

| Colour | Meaning |
|---|---|
| 🟢 Green outline | Free parking slot |
| 🔴 Red outline | Occupied parking slot |
| 🟡 Yellow outline | Nearest free slot (A* target) |
| 🟠 Orange polyline | A* path from entry to nearest free slot |
| 🟣 Magenta dot | Entry point; also projected bottom-center detection point |
| 🔵 Cyan dot | Projected bbox-center detection point (polygon slot test) |
| 🔵 Cyan outline | Polygon slot boundary |
| 🔵 Cyan rectangle | YOLO detected vehicle bounding box (in debug window) |
| 🔴 Dark red fill | User-defined obstacle / no-go zone |
| `LIVE` (green) | YOLO inference ran on this frame |
| `CACHED` (yellow) | Frame-skip cache used — no inference this frame |

---

## Results

### Detection Pipeline Comparison (COCO val2017 reference)

| Model | Params (M) | mAP50-95 | mAP50 | FPS (T4 GPU) |
|---|---|---|---|---|
| Classical CV | — | — | — | ~200+ (CPU) |
| YOLOv8n | 3.2 | 37.3% | 52.5% | ~300 |
| YOLOv8s | 11.2 | 44.9% | 61.8% | ~200 |
| RT-DETR-R50 | 42 | 53.1% | 71.3% | ~108 |
| D-FINE-L | 31 | 54.0% | 71.8% | ~124 |
| D-FINE-X | 62 | 55.8% | 73.5% | ~78 |

> Slot-level Precision / Recall / F1 on your specific video requires running `evaluate.py` with a labelled ground-truth CSV.

### Estimated per-source slot accuracy

| Pipeline | Precision | Recall | F1 | FPS (approx) |
|---|---|---|---|---|
| Classical CV | 0.85–0.90 | 0.80–0.88 | 0.83–0.89 | 150–200 |
| YOLOv8n, skip=5 | 0.90–0.94 | 0.88–0.93 | 0.89–0.93 | 35–50 |
| YOLOv8s, skip=5 | 0.93–0.96 | 0.91–0.95 | 0.92–0.95 | 25–35 |

---

## Quick Start

```bash
# 1. Add a video source (no X/Y prompt — set entry point visually with 'e')
python mark_parking_slots.py --add
# OR
python config.py --add

# 2. Mark parking slots interactively
python mark_parking_slots.py --source <source_id>
# On first run: choose crop / calibrate / skip in the setup menu
# Press 'p' to toggle polygon slot mode for angled CCTV views
# Press 'e' then click to place the entry point
# Press 's' or 'q' to save

# 3a. Run classical CV pipeline
python detect_improved.py --source <source_id>

# 3b. Run YOLO pipeline
python detect_yolo.py --source <source_id> --model yolov8n.pt --skip 5

# 4. Start web GUI (live stream + Classical ↔ YOLO mode switch)
python app.py --port 8000
# Open http://localhost:8000 in browser

# 5. Evaluate and compare pipelines
python evaluate.py --source <source_id> --ground-truth gt.csv --output report.csv
```

---

## Requirements

```
ultralytics==8.4.54
opencv-python==4.13.0.92
numpy==2.4.4
fastapi==0.136.3
uvicorn==0.48.0
python-multipart==0.0.29
pytest==9.0.3
```

Install: `pip install -r requirements.txt`

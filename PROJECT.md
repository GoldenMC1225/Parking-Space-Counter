# Parking Space Counter

A real-time parking lot occupancy detection system that supports multiple video sources, two detection pipelines, perspective correction, A* guided navigation, and a web-based monitoring interface.

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
apply_crop()          ← optional ROI crop (stored in slots JSON)
    │
    ▼
warpPerspective(H)    ← optional homography correction (bird's-eye view)
    │
    ├─── Classical CV path ──────────────────────────────────────────────
    │    convert_grayscale() → check_slots() → statuses (True = free)
    │
    └─── YOLO path ──────────────────────────────────────────────────────
         detect on original frame → filter_detections()
             → project_boxes_to_warped(H)
             → assign_slots() → statuses (True = occupied)
    │
    ▼
find_nearest_free_slot()   ← Euclidean search over free slots
    │
    ▼
build_obstacle_grid()      ← occupied slots + user obstacles → grid
    │
    ▼
astar(grid, entry, goal)   ← 8-directional A* pathfinding
    │
    ▼
draw_slots()               ← annotate frame with colours, path, HUD
    │
    ▼
Display (OpenCV window) or serve via MJPEG stream (FastAPI)
```

### Detection Pipelines

| Pipeline | File | Method |
|---|---|---|
| Classical CV | `detect_improved.py` | Grayscale threshold + contour pixel count |
| YOLO | `detect_yolo.py` | YOLOv8/RT-DETR/D-FINE + slot assignment |

Classical CV is fast, runs on CPU, but sensitive to lighting and shadows.
YOLO is robust to lighting variation and distinguishes vehicle types, but requires a GPU for real-time speed.

### Slot Assignment Logic (YOLO)

Detection runs on the original (side-view) frame where YOLO was trained. Bottom-center and bbox-center points are projected through the homography matrix into warped space, then tested against slot boundaries:

- **Rectangle slots** — bounding-box containment test
- **Polygon slots** — `cv2.pointPolygonTest` (handles angled CCTV views)

Both bbox-center and bottom-center are tested; a slot is occupied if either point lands inside it.

---

## File Structure

```
├── config.py               Data models + JSON load/save helpers
├── utils.py                A*, obstacle grid, nearest-slot search, draw_slots
├── mark_parking_slots.py   Interactive slot marking tool (OpenCV GUI)
├── detect_improved.py      Classical CV detection pipeline
├── detect_yolo.py          YOLO detection pipeline
├── app.py                  FastAPI web server (MJPEG stream, mode switch)
├── evaluate.py             Precision / Recall / FPS evaluation framework
├── sources.json            List of video/RTSP sources
└── slots/
    └── <source_id>.json    Slot coordinates, homography, obstacles per source
```

---

## Important Functions

### `utils.py`

| Function | Purpose |
|---|---|
| `astar(grid, start_px, goal_px)` | 8-directional A* on a 20×20 px cell grid. Returns a list of pixel waypoints or `None` if unreachable. |
| `build_obstacle_grid(frame_shape, cfg, statuses)` | Rasterises occupied slots and user-drawn obstacle polygons into a 2-D numpy grid for A*. |
| `find_nearest_free_slot(entry, cfg, statuses)` | Returns the index of the free slot with the smallest Euclidean distance to the entry point. Covers both rect and polygon slots. |
| `draw_slots(frame, cfg, statuses, ...)` | Draws slot rectangles/polygons (green/red/yellow), A* path (orange), entry point (magenta), obstacle fills, and the HUD onto the frame. |
| `apply_crop(frame, cfg)` | Crops the frame to `cfg.crop_region` `[x, y, w, h]` when set; returns the frame unchanged otherwise. |

### `detect_yolo.py`

| Function | Purpose |
|---|---|
| `filter_detections(raw_results, conf_thresh)` | Keeps only detections whose class is in `{car, truck, bus, motorcycle}` and confidence ≥ threshold. |
| `project_boxes_to_warped(boxes, H)` | Projects both bbox-center and bottom-center of each detection through homography H into warped coordinates. |
| `assign_slots(boxes, cfg)` | Tests each detection's projected center/bottom-center against every slot boundary. Returns `List[bool]` (True = occupied). |
| `bottom_center(x1, y1, x2, y2)` | Returns `(int((x1+x2)/2), int(y2))` — the bottom edge midpoint of a bounding box. |
| `run(source, model_path, skip, conf)` | Full YOLO frame loop: crop → detect on original → project → assign → warp for display → A* → draw → show. |

### `mark_parking_slots.py`

| Function | Purpose |
|---|---|
| `interactive_crop(frame)` | Drag-to-select crop region on the full frame. Returns `[x, y, w, h]` or `None`. |
| `run_calibration(frame, w, h)` | Full homography calibration state machine: collect 4 points → collinearity check → compute H → show preview → accept/redo. |
| `collect_calibration_points(frame)` | Mouse callback that collects exactly 4 click points for homography calibration. |
| `compute_homography(src_pts, w, h)` | Wraps `cv2.getPerspectiveTransform` to map 4 source points to the 4 corners of a `w×h` canvas. |
| `main(source)` | Interactive slot-marking loop. Supports mode toggle (rect/poly), obstacle drawing, entry point placement, frame navigation, and auto-save. |

### `app.py`

| Function | Purpose |
|---|---|
| `_classical_pipeline_loop(state)` | Background thread: reads frames, applies crop+warp, runs classical CV detection, updates `state.latest_frame` and `state.status`. |
| `_yolo_pipeline_loop(state, ...)` | Background thread: reads frames, runs YOLO with frame-skip cache, updates shared state. |
| `switch_pipeline(state, new_mode)` | Stops the current thread (3 s timeout), starts a new thread in the requested mode. `latest_frame` is preserved during transition. |
| `stream(source_id)` | FastAPI endpoint — streams MJPEG (`multipart/x-mixed-replace`) from `state.latest_frame`. |
| `status(source_id)` | Returns JSON: `free_slots`, `total_slots`, `nearest_free_slot`, `fps`, `mode`, `source_name`. |

### `config.py`

| Class/Function | Purpose |
|---|---|
| `SlotConfig` | Dataclass holding rect slots, polygon slots, homography matrix, warped size, crop region, obstacle polygons, and detection threshold. |
| `Source` | Dataclass for a video/RTSP source: id, name, URI, slots file path, entry point. |
| `load_slots(source)` | Deserialises `SlotConfig` from the source's JSON file. Defaults to empty config if file missing. |
| `save_slots(source, cfg)` | Serialises `SlotConfig` back to JSON (creates directories if needed). |

---

## Slot Marking Tool Controls

```
p          → toggle RECTANGLE / POLYGON slot mode
Left-click → add slot (rect: drag to set size; polygon: click 4 corners)
Right-click→ undo last slot of current type
z          → undo last polygon slot
r          → clear all rectangle slots
e          → set entry point (click to place)
o          → draw obstacle polygon (Enter = save, Esc = cancel)
x          → undo last obstacle polygon
h          → recalibrate homography
n          → advance to next frame
s          → save
q          → quit (auto-saves)
```

---

## Visual Legend

| Colour | Meaning |
|---|---|
| 🟢 Green outline | Free parking slot |
| 🔴 Red outline | Occupied parking slot |
| 🟡 Yellow outline | Nearest free slot (A* target) |
| 🟠 Orange polyline | A* path from entry to nearest free slot |
| 🟣 Magenta dot | Entry point / bottom-center detection point |
| 🔵 Cyan dot | Bbox center detection point (polygon slot test) |
| 🔵 Cyan rectangle | YOLO detected vehicle bounding box |
| 🔴 Dark red fill | User-defined obstacle / no-go zone |

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
# 1. Add a video source
python config.py --add

# 2. Mark parking slots interactively
python mark_parking_slots.py --source <source_id>

# 3a. Run classical CV pipeline
python detect_improved.py --source <source_id>

# 3b. Run YOLO pipeline
python detect_yolo.py --source <source_id> --model yolov8n.pt --skip 5

# 4. Start web GUI (live stream + mode switch)
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

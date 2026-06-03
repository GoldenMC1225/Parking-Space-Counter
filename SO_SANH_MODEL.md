# So sánh mô hình — Dự án Parking Space Counter

## Tổng quan hệ thống

Dự án có **3 pipeline** phát hiện xe:

| Pipeline | File | Phương pháp |
|---|---|---|
| **Classical CV** | `detect_improved.py` | Grayscale → Threshold → Đếm pixel contour |
| **YOLOv8** | `detect_yolo.py` | YOLO inference → bottom-center point → slot assignment |
| **RT-DETR / D-FINE** | *(đề xuất thay thế YOLO)* | Transformer-based detection |

Đánh giá thực tế chạy qua `evaluate.py` với ground-truth CSV, đo **Precision, Recall, F1, FPS** trên từng frame của `video/CarPark.mp4`.

---

## 1. So sánh chỉ số — COCO val2017 (benchmark tham khảo)

> Đây là chỉ số trên tập COCO chuẩn. Chỉ số thực tế trên `CarPark.mp4` sẽ khác — cần chạy `evaluate.py` để có số liệu chính xác cho dự án này.

| Model | Params (M) | mAP50-95 | mAP50 | FPS (T4 GPU) | Ghi chú |
|---|---|---|---|---|---|
| **Classical CV** | — | — | — | ~200+ | Không dùng model, chạy CPU |
| **YOLOv8n** | 3.2 | 37.3% | 52.5% | ~300 | Default trong `detect_yolo.py` |
| **YOLOv8s** | 11.2 | 44.9% | 61.8% | ~200 | `--model yolov8s.pt` |
| **YOLOv8m** | 25.9 | 50.2% | 67.2% | ~120 | `--model yolov8m.pt` |
| **RT-DETR-R18** | 20 | 46.5% | 63.8% | 217 | Nhẹ nhất trong RT-DETR |
| **RT-DETR-R50** | 42 | 53.1% | 71.3% | 108 | Cân bằng tốt |
| **RT-DETRv2-R50** | 42 | 53.4% | — | 108 | Cải tiến từ RT-DETR |
| **D-FINE-S** | **10** | **48.5%** | **65.5%** | **287** | Nhỏ + nhanh nhất |
| **D-FINE-M** | **19** | **52.3%** | **69.8%** | **180** | Cân bằng tốt nhất |
| **D-FINE-L** | **31** | **54.0%** | **71.8%** | **124** | Khuyến nghị cho dự án |
| **D-FINE-X** | **62** | **55.8%** | **73.5%** | **78** | Chính xác nhất |

---

## 2. Giải thích chỉ số trong ngữ cảnh dự án

### Precision (P) — Độ chính xác
```
P = TP / (TP + FP)
```
- **TP**: Slot dự đoán *có xe* và thực tế *có xe*
- **FP**: Slot dự đoán *có xe* nhưng thực tế *trống*
- Precision thấp → báo sai nhiều slot bận → người dùng đến chỗ trống nhưng thấy bị đánh dấu bận

### Recall (R) — Độ bao phủ
```
R = TP / (TP + FN)
```
- **FN**: Slot dự đoán *trống* nhưng thực tế *có xe*
- Recall thấp → bỏ sót xe → hướng dẫn người dùng đến slot đã có xe

### F1-Score
```
F1 = 2 × P × R / (P + R)
```
- Trung bình điều hòa của P và R
- Chỉ số tổng hợp quan trọng nhất cho bài toán này

### FPS — Tốc độ xử lý
- Classical CV: ~200 FPS (CPU, không cần GPU)
- YOLO với `--skip 5`: chỉ inference 1/5 frame → FPS hiển thị cao hơn thực tế
- Đo thực tế trong `evaluate.py`: `avg_fps = evaluated_frames / total_wall_time`

### mAP (tham khảo COCO)
- **mAP50**: Bbox đúng nếu IoU ≥ 0.50 với ground-truth
- **mAP50-95**: Trung bình tại IoU từ 0.50 → 0.95 (nghiêm ngặt hơn)
- Trong dự án này, chỉ số quan trọng hơn là **Precision/Recall trên slot** (không phải bbox), vì `assign_slots` dùng bottom-center point chứ không dùng IoU

---

## 3. So sánh pipeline trong dự án

### Classical CV (`detect_improved.py`)

```
Frame → convert_grayscale() → check_slots() → statuses (True=free)
```

**Ưu điểm:**
- Không cần GPU, chạy trên mọi máy
- FPS rất cao (~200+)
- Không cần training data

**Nhược điểm:**
- Nhạy với thay đổi ánh sáng, bóng đổ, vết bẩn
- `threshold=30` cần chỉnh thủ công theo từng camera
- Không phân biệt được xe với vật thể khác

**Điểm yếu cụ thể với `CarPark.mp4`:**
- Góc camera nghiêng → bóng xe che slot bên cạnh → false positive
- Ánh sáng thay đổi theo thời gian → threshold cố định không còn phù hợp

---

### YOLO Pipeline (`detect_yolo.py`)

```
Frame → warp_frame(H) → YOLO inference → filter_detections()
      → assign_slots() → statuses (True=occupied)
```

**Ưu điểm:**
- Robust với ánh sáng, bóng đổ
- Phân biệt được xe với người, xe đạp, v.v.
- Homography warp → bird's-eye view → bottom-center point chính xác hơn
- Frame-skip caching (`--skip 5`) → tiết kiệm CPU/GPU

**Nhược điểm:**
- Cần GPU để đạt FPS cao
- YOLOv8n (default) có mAP thấp hơn RT-DETR/D-FINE
- Cần fine-tune trên dataset bãi đỗ xe để đạt P/R cao

**Tham số quan trọng:**
```bash
python detect_yolo.py --model yolov8n.pt --skip 5 --conf 0.25
```
- `--skip 5`: inference mỗi 5 frame → giảm tải GPU 5×
- `--conf 0.25`: ngưỡng confidence → tăng lên 0.4 để giảm FP

---

### RT-DETR / D-FINE (đề xuất thay thế)

Để tích hợp vào dự án, thay `YOLO` bằng RT-DETR/D-FINE trong `detect_yolo.py`:

```python
# Hiện tại (detect_yolo.py)
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Thay bằng RT-DETR (ultralytics hỗ trợ sẵn)
model = YOLO("rtdetr-r18.pt")   # RT-DETR-R18
model = YOLO("rtdetr-r50.pt")   # RT-DETR-R50

# Hoặc D-FINE (cần cài thêm)
# pip install dfine
```

**Lý do D-FINE-L là lựa chọn tốt nhất cho dự án:**
- 31M params — nhỏ hơn RT-DETR-R101 (76M) nhưng mAP tương đương
- 124 FPS — đủ nhanh cho real-time với `--skip 3`
- mAP50 = 71.8% — cao hơn YOLOv8m (67.2%) với params tương đương

---

## 4. Kết quả đánh giá thực tế (chạy `evaluate.py`)

Để có số liệu P, R, F1, FPS thực tế trên `CarPark.mp4`:

```bash
# Bước 1: Tạo ground-truth CSV (gán nhãn thủ công)
# Format: frame_index,slot_id,occupied
# Ví dụ: 0,0,1  (frame 0, slot 0, có xe)

# Bước 2: Chạy đánh giá
python evaluate.py \
  --source carpark_main \
  --ground-truth ground_truth.csv \
  --output evaluation_report.csv
```

Output CSV có dạng:
```
frame,pipeline,precision,recall,f1_score,fps
0,classical,0.92,0.88,0.90,187.3
0,yolo,0.96,0.94,0.95,42.1
```

**Bảng kết quả mẫu (ước tính dựa trên đặc điểm video):**

| Pipeline | P (ước tính) | R (ước tính) | F1 (ước tính) | FPS thực tế |
|---|---|---|---|---|
| Classical CV | ~0.85–0.90 | ~0.80–0.88 | ~0.83–0.89 | ~150–200 |
| YOLOv8n (skip=5) | ~0.90–0.94 | ~0.88–0.93 | ~0.89–0.93 | ~35–50 |
| YOLOv8s (skip=5) | ~0.93–0.96 | ~0.91–0.95 | ~0.92–0.95 | ~25–35 |

> Số liệu thực tế phụ thuộc vào chất lượng homography calibration và ground-truth CSV.

---

## 5. A* Pathfinding — Tìm đường đến slot gần nhất

### Nguyên lý trong `utils.py`

```
f(n) = g(n) + h(n)
```
- `g(n)`: Chi phí thực tế từ ENTRY_POINT đến ô n (tích lũy)
- `h(n)`: Euclidean distance từ n đến đích (heuristic)
- Lưới: `GRID_CELL = 20` pixel/ô
- Slot có xe → ô vật cản (`grid[y][x] = 1`)

### Luồng xử lý trong dự án

```
statuses (True=free) 
    → find_nearest_free_slot()   # Euclidean gần nhất
    → build_obstacle_grid()      # Slot bận = obstacle
    → astar(grid, entry, goal)   # Tìm đường
    → draw_slots(..., path)      # Vẽ đường cam lên frame
```

### Màu sắc hiển thị

| Màu | Ý nghĩa |
|---|---|
| 🟢 Xanh lá | Slot trống |
| 🔴 Đỏ | Slot có xe |
| 🟡 Vàng | Slot trống gần nhất (đích A*) |
| 🟠 Cam | Đường A* từ ENTRY đến đích |
| 🟣 Tím | Điểm vào (ENTRY_POINT) |

### Cải tiến A* đề xuất

**a) Weighted A* — tránh khu vực đông xe:**
```python
# Trong astar(), thêm cost từ mật độ xe xung quanh
density_cost = count_nearby_obstacles(grid, node, radius=3)
f = g + h + 0.5 * density_cost
```

**b) Multi-goal A* — tìm slot gần nhất theo đường đi thực tế:**
```python
# Thay vì chọn slot gần nhất theo Euclidean rồi mới chạy A*
# Chạy Dijkstra từ entry → dừng khi gặp slot trống đầu tiên
# Đảm bảo slot "gần nhất" theo đường đi thực tế, không phải đường chim bay
```

**c) Tích hợp detection boxes vào obstacle grid:**
```python
# Thêm bounding box của xe phát hiện được vào grid
# (không chỉ dùng slot rectangles)
# → Đường A* tránh được thân xe, không chỉ tránh slot
def build_obstacle_grid_with_detections(frame_shape, cfg, statuses, boxes):
    grid = build_obstacle_grid(frame_shape, cfg, statuses)
    for x1, y1, x2, y2, *_ in boxes:
        gx1, gy1 = x1 // GRID_CELL, y1 // GRID_CELL
        gx2, gy2 = x2 // GRID_CELL, y2 // GRID_CELL
        grid[gy1:gy2+1, gx1:gx2+1] = 1
    return grid
```

---

## 6. Kết luận và khuyến nghị

| Tiêu chí | Classical CV | YOLOv8n | D-FINE-L |
|---|---|---|---|
| Độ chính xác | Trung bình | Tốt | Tốt nhất |
| Tốc độ (CPU) | ✅ Rất nhanh | ❌ Chậm | ❌ Chậm |
| Tốc độ (GPU) | ✅ Rất nhanh | ✅ Nhanh | ✅ Nhanh |
| Robust ánh sáng | ❌ Kém | ✅ Tốt | ✅ Tốt |
| Cần training | ❌ Không | ⚠️ Fine-tune | ⚠️ Fine-tune |
| Tích hợp hiện tại | ✅ Có sẵn | ✅ Có sẵn | 🔧 Cần thêm |

**Khuyến nghị:**
- **Môi trường không có GPU**: Dùng Classical CV + tăng cường preprocessing (CLAHE, adaptive threshold)
- **Có GPU, cần nhanh**: YOLOv8s với `--skip 3 --conf 0.35`
- **Cần độ chính xác cao nhất**: D-FINE-L với homography calibration đầy đủ
- **Báo cáo học thuật**: Chạy `evaluate.py` để có P/R/F1/FPS thực tế, so sánh 3 pipeline trên cùng ground-truth CSV

---

## 7. Lệnh chạy nhanh

```bash
# Classical CV
python detect_improved.py --source carpark_main

# YOLO (YOLOv8n, mặc định)
python detect_yolo.py --source carpark_main

# YOLO với model tốt hơn
python detect_yolo.py --source carpark_main --model yolov8s.pt --skip 3 --conf 0.35

# RT-DETR (qua ultralytics)
python detect_yolo.py --source carpark_main --model rtdetr-r50.pt --skip 1

# Đánh giá so sánh
python evaluate.py --source carpark_main --ground-truth gt.csv --output report.csv

# Web GUI (xem cả 2 pipeline, switch real-time)
python app.py --port 8000
```

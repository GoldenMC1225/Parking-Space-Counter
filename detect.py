import cv2
import numpy as np
import time

video_path = "./video/CarPark.mp4"
cap = cv2.VideoCapture(video_path)
parking_slots = [(402, 239), (753, 377), (55, 100), (56, 146), (51, 241), (53, 290), (51, 192), (405, 189), (402, 138), (405, 90), (514, 92), (511, 139), (514, 187), (512, 236), (163, 99), (164, 147), (158, 194), (159, 243), (161, 290), (55, 337), (162, 339), (160, 388), (162, 429), (52, 431), (53, 479), (163, 479), (168, 525), (165, 576), (165, 620), (56, 623), (51, 573), (52, 527), (402, 289), (402, 338), (404, 382), (405, 427), (405, 526), (403, 569), (406, 619), (512, 524), (512, 568), (513, 620), (511, 426), (511, 380), (513, 329), (511, 284), (751, 88), (751, 136), (750, 188), (753, 232), (753, 276), (751, 327), (757, 427), (753, 472), (757, 518), (760, 573), (760, 616), (901, 620), (901, 576), (892, 141), (892, 190), (893, 235), (894, 284), (897, 330), (898, 375), (901, 424), (903, 474), (899, 522), (46, 385)]
rect_width, rect_height = 100, 33
color = (0,0,255)
thick = 1
threshold = 30
last_call_time = time.time()
prevFreeslots=0


def convert_grayscale(frame):
    # Chuyển đổi hình ảnh sang thang độ xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Áp dụng ngưỡng để tạo hình ảnh nhị phân (chỉ có đen và trắng)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Tìm các đường viền trong hình ảnh nhị phân
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo một canvas màu đen với cùng kích thước như hình ảnh đầu vào
    contour_image = frame.copy()
    contour_image[:] = 0  # Điền màu đen

    # Vẽ các đường viền trên canvas đen bằng màu trắng
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=2)
    return contour_image

def mark_slots(frame, grayscale_frame):
    global last_call_time
    global prevFreeslots
    current_time = time.time()
    elapsed_time = current_time - last_call_time

    # Lấy kích thước frame để kiểm tra giới hạn
    frame_height, frame_width = frame.shape[:2]

    freeslots=0
    for x, y in parking_slots:
        x1=x+10
        x2=x+rect_width-11
        y1=y+4
        y2=y+rect_height
        start_point, stop_point = (x1,y1), (x2, y2)

        # Kiểm tra xem vùng cắt có nằm trong giới hạn frame không
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            # Nếu vượt quá, bỏ qua slot này hoặc điều chỉnh tọa độ
            continue
        
        # Kiểm tra xem vùng cắt có hợp lệ không (không rỗng)
        if x2 <= x1 or y2 <= y1:
            continue

        try:
            crop=grayscale_frame[y1:y2, x1:x2]
            
            # Kiểm tra xem crop có rỗng không
            if crop.size == 0:
                continue
            
            # Kiểm tra số kênh màu của crop
            if len(crop.shape) == 3:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray_crop = crop

            # Đếm số pixel khác không (pixel trắng) trong vùng cắt
            count=cv2.countNonZero(gray_crop)

            # Gán màu và độ dày dựa trên ngưỡng: xanh lá nếu trống, đỏ nếu có xe
            color, thick = [(0,255,0), 5] if count<threshold else [(0,0,255), 2]

            if count<threshold:
                freeslots = freeslots+1
            
            cv2.rectangle(frame, start_point, stop_point, color, thick)
        except Exception as e:
            # Bỏ qua slot nếu có lỗi và tiếp tục với slot tiếp theo
            print(f"Lỗi khi xử lý slot tại ({x}, {y}): {e}")
            continue

        ##  hiển thị số lượng pixel khác không trong mỗi hình chữ nhật chỗ đỗ xe
        text_x = x1+5
        text_y = y1 + 10  # Điều chỉnh tọa độ Y để đặt văn bản phía trên hình chữ nhật
        cv2.putText(frame, str(count), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 255), 1)

    # Cập nhật bộ đếm hiển thị số chỗ trống - ít thường xuyên hơn để tránh nhấp nháy
    current_time = time.time()
    if current_time - last_call_time >= 0.1:
        cv2.putText(frame, "Free Slots:" + str(freeslots), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2)
        last_call_time = current_time
        prevFreeslots = freeslots
    else:
         cv2.putText(frame, "Free Slots:" + str(prevFreeslots), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2)
    return frame
    
while True:

        # Đọc video từng khung hình một
        ret, frame = cap.read()

        if not ret:
            # Nếu hết video, quay lại đầu video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        try:
            grayscale_frame = convert_grayscale(frame)
            out_image = mark_slots(frame, grayscale_frame)        
            
            # Hiển thị kết quả
            cv2.imshow("Parking Spot Detector", out_image)
        except Exception as e:
            print(f"Lỗi khi xử lý frame: {e}")
            continue
    
        # Điều kiện thoát: nhấn phím 'q'
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
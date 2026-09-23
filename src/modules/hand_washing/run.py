from collections import deque

from ultralytics import YOLO
import cv2
import numpy as np
from draw import DrawImage, select_roi_points

# Ngưỡng confidence tối thiểu để tin vào keypoint wrist (tránh false positive khi bị che khuất)
WRIST_CONF_THRESHOLD = 0.5
# Ngưỡng khoảng cách 2 cổ tay (pixel) để coi là đang chụm tay lại
WRIST_DIST_THRESHOLD = 100
# Số frame gần nhất dùng để làm mượt (smoothing) trạng thái, tránh nhấp nháy
SMOOTH_WINDOW = 15
# Tỉ lệ frame "Washing" trong SMOOTH_WINDOW để chốt trạng thái hiển thị là Washing
SMOOTH_RATIO = 0.6


# MARK: PROCESS
def process_camera(frame, draw_image, model, window_name, status_buffer):
    status = "Nothing"
    h, w = frame.shape[:2]
    x_start = w // 3
    x_end = ( w // 3 ) * 2

    draw_image.draw_center(frame)
    draw_image.draw_region(frame, x_start, 0, x_end, h)
    draw_image.draw_roi(frame)

    # Chỉ crop vùng giữa để đưa vào model, frame gốc giữ nguyên để hiển thị
    crop = frame[:, x_start:x_end]

    results = model.predict(crop, conf=0.5, verbose=False)
    if results[0].keypoints is None or len(results[0].boxes) == 0:
        status_buffer.append(0)
        cv2.imshow(window_name, frame)
        return status

    boxes = results[0].boxes.xyxy.cpu().numpy()
    keypoints = results[0].keypoints.xy.cpu().numpy()
    keypoints_conf = results[0].keypoints.conf.cpu().numpy()

    # Chọn người có box lớn nhất (gần camera nhất) để lấy keypoint wrist
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))

    target_hand = keypoints[best_idx]
    target_hand_conf = keypoints_conf[best_idx]

    # Cộng lại x_start để quy đổi tọa độ từ ảnh crop về frame gốc
    right_x, right_y = target_hand[10]
    right_conf = target_hand_conf[10]
    right_x += x_start

    left_x, left_y = target_hand[9]
    left_conf = target_hand_conf[9]
    left_x += x_start

    is_washing_this_frame = 0

    if right_conf > WRIST_CONF_THRESHOLD and left_conf > WRIST_CONF_THRESHOLD:
        rText = f"Right:{right_conf:.2f}"
        lText = f"Left:{left_conf:.2f}"

        draw_image.draw_wrist_point(frame, right_x, right_y, 5, 0, 0, 255, rText)
        draw_image.draw_wrist_point(frame, left_x, left_y, 5, 0, 255, 0, lText)

        dist = np.sqrt(
            (right_x - left_x) * (right_x - left_x)
            + (right_y - left_y) * (right_y - left_y)
        )

        print(dist)
        right_in_roi = draw_image.is_point_in_roi(right_x, right_y)
        left_in_roi = draw_image.is_point_in_roi(left_x, left_y)

        is_washing_this_frame = int(
            dist < WRIST_DIST_THRESHOLD and right_in_roi and left_in_roi
        )

    status_buffer.append(is_washing_this_frame)

    # Chốt trạng thái hiển thị dựa trên tỉ lệ frame "Washing" trong cửa sổ gần nhất
    # để tránh nhấp nháy do keypoint dao động nhẹ giữa các frame liên tiếp.
    if len(status_buffer) > 0 and sum(status_buffer) / len(status_buffer) >= SMOOTH_RATIO:
        status = "Washing"
    else:
        status = "Not Washing"

    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(window_name, frame)
    return status


# MARK: RUN
model = YOLO("yolo26n-pose.pt")

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Đọc thử 1 frame trước khi khởi tạo DrawImage: với CAP_DSHOW trên Windows,
# cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT) có thể chưa phản ánh đúng resolution
# vừa set nếu đọc ngay lập tức, dẫn đến ROI/center bị tính sai kích thước.
ret0, frame0 = cap1.read()
if not ret0:
    raise RuntimeError("Không đọc được frame từ camera, kiểm tra lại kết nối camera.")

actual_h, actual_w = frame0.shape[:2]
print(f"Camera thực tế: {actual_w}x{actual_h}")

draw_image1 = DrawImage(cap1, "cam2")

# Cho người dùng tự chọn ROI bằng cách click 2 điểm trên frame đầu tiên,
# nếu không chọn (bấm 'q') thì giữ nguyên ROI mặc định tính theo tỷ lệ trong DrawImage.
roi_points = select_roi_points(cap1, center=(draw_image1.xCenter, draw_image1.yCenter))
if roi_points is not None:
    draw_image1.roiX1, draw_image1.roiY1, draw_image1.roiX2, draw_image1.roiY2 = roi_points
    print(f"ROI đã chọn: {roi_points}")
else:
    print("Giữ ROI mặc định (không chọn thủ công).")

status_buffer1 = deque(maxlen=SMOOTH_WINDOW)

max_consecutive_fail = 30
fail_count = 0

while True:
    ret1, frame1 = cap1.read()

    if not ret1:
        fail_count += 1
        print(f"Không đọc được frame ({fail_count}/{max_consecutive_fail})")
        if fail_count >= max_consecutive_fail:
            print("Mất kết nối camera, dừng chương trình.")
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    fail_count = 0
    status1 = process_camera(frame1, draw_image1, model, "Camera 1 - Webcam", status_buffer1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cv2.destroyAllWindows()

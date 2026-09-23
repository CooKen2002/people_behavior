import cv2
import numpy as np


def select_roi_points(cap, center=None, window_name="Select ROI - click 2 points, q to cancel"):
    """
    Camera chạy realtime (đọc frame liên tục mỗi vòng lặp) trong lúc chờ người dùng
    click 2 điểm (top-left, bottom-right) để tự định nghĩa vùng ROI, thay vì tính
    cứng theo tỷ lệ hoặc đứng yên trên 1 frame tĩnh.
    cap: cv2.VideoCapture đang mở.
    center: tọa độ (x, y) điểm center để vẽ tham chiếu (vòng tròn đỏ), tùy chọn.
    Trả về (x1, y1, x2, y2) đã sắp xếp đúng thứ tự, hoặc None nếu người dùng bấm 'q' thoát sớm
    hoặc không đọc được frame từ camera.
    """
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # Không đọc được frame mới: dùng lại frame gần nhất nếu có, tránh crash
            if last_frame is None:
                cv2.destroyWindow(window_name)
                return None
            frame = last_frame
        else:
            last_frame = frame

        display = frame.copy()

        if center is not None:
            cv2.circle(display, (int(center[0]), int(center[1])), 10, (0, 0, 255), 5)

        for p in points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.rectangle(display, points[0], points[1], (255, 0, 255), 2)

        cv2.putText(
            display,
            "Click 2 diem (top-left, bottom-right) - q: huy",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        # Đã chọn đủ 2 điểm: giữ khung hình cuối hiển thị 1 chút rồi thoát vòng lặp
        if len(points) == 2:
            cv2.waitKey(500)
            break

        # Người dùng bấm 'q': huỷ chọn, thoát vòng lặp ngay
        if key == ord("q"):
            break

    cv2.destroyWindow(window_name)

    if len(points) < 2:
        return None

    (x1, y1), (x2, y2) = points
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


class DrawImage():
    def __init__(self, cap, cam_id):
        self.wImage = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.hImage = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if cam_id == "cam1": 
            self.xCenter = int(self.wImage/3)
            self.yCenter = int(self.hImage/3)
        elif cam_id == "cam2": 
            self.xCenter = int(self.wImage/3*2)
            self.yCenter = int(self.hImage/3)
        else:
            self.xCenter = int(self.wImage/2)
            self.yCenter = int(self.hImage/4)
        
        # ROI (vùng bồn rửa) đặt quanh xCenter/yCenter:
        # - Trục X: căn giữa (roiX1/roiX2 đối xứng qua xCenter)
        # - Trục Y: dịch xuống dưới (chỉ lùi lên 1/4 roi_h nhưng mở xuống dưới full roi_h)
        #   vì bồn rửa/tay thường ở dưới điểm center, không đối xứng qua yCenter
        roi_w = self.wImage * 0.35  # roi_w_ratio=0.35, roi_h_ratio=0.5
        roi_h = self.hImage * 0.5
        self.roiX1 = int(self.xCenter - roi_w / 2)
        self.roiY1 = int(self.yCenter - roi_h / 4)
        self.roiX2 = int(self.xCenter + roi_w / 2)
        self.roiY2 = int(self.yCenter + roi_h)

    def draw_center(self, frame):
        cv2.circle(frame, (self.xCenter, self.yCenter), 10, (0, 0, 255), 5)
    
    def draw_region(self, frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

    def draw_roi(self, frame, color=(255, 0, 255), thickness=2, text="ROI"):
        cv2.rectangle(
            frame, (self.roiX1, self.roiY1), (self.roiX2, self.roiY2), color, thickness
        )
        if text is not None:
            cv2.putText(
                frame,
                str(text),
                (self.roiX1, self.roiY1 - 10 if self.roiY1 - 10 > 10 else self.roiY1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def is_point_in_roi(self, x, y):
        return self.roiX1 <= x <= self.roiX2 and self.roiY1 <= y <= self.roiY2

    def draw_wrist_point(self, frame, x, y, radius, r, g, b, text):
        cv2.circle(frame, (int(x), int(y)), int(radius), (int(r), int(g), int(b)), -1)
        cv2.putText(
            frame,
            str(text),
            (int(x) + 15, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (r, g, b),
            2,
        )
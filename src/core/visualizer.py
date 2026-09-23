import cv2
class DrawImage():
    def __init__(self, cap, cam_id):
        self.wImage = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.hImage = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
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
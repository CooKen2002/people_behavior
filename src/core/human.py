import time
from typing import List, Tuple, Dict, Any, Optional

class Human:
    def __init__(self, track_id: str, bbox: List[float]):
        self.track_id = track_id
        self.bbox = bbox if bbox is not None else []    # [x1, y1, x2, y2]
        self.hits = 1                                   # Đếm tổng số frame tracking liên tục
        self.time_since_update = 0                         # Đếm số frame không thấy update (để xóa track cũ)

    def update(self, bbox: List[float]):
        """Cập nhật tọa độ bbox mới và reset thời gian vắng mặt"""
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self):
        """Đánh dấu 1 frame không tìm thấy match với object này"""
        self.time_since_update += 1

    def check_iou_with_roi(self, roi_bbox: List[float]) -> float:
        """Tính toán Intersection over Union (IoU) giữa bounding box của con người và ROI."""
        human_x1, human_y1, human_x2, human_y2 = self.bbox
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox

        inter_x1 = max(human_x1, roi_x1)
        inter_y1 = max(human_y1, roi_y1)
        inter_x2 = min(human_x2, roi_x2)
        inter_y2 = min(human_y2, roi_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        human_area = (human_x2 - human_x1) * (human_y2 - human_y1)
        roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)

        union_area = human_area + roi_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou
    
class HumanBehavior(Human):
    def __init__(self, track_id: str, keypoints: Optional[List[List[float]]] = None, current_time: float = None, **kwargs):
        super().__init__(track_id=track_id, **kwargs)
        self.keypoints = keypoints
        self.current_time = current_time

        # --- 1. Thông tin chung
        self.first_seen = current_time if current_time is not None else time.time()  # Thời điểm xuất hiện lần đầu (timestamp)
        self.last_seen = self.first_seen  # Thời điểm cập nhật khung hình gần nhất
        self.keypoints = keypoints if keypoints is not None else []  # Tọa độ các khớp xương (nếu dùng Pose Estimation)

        # --- 2. Trạng thái hành vi (Behavioral State) ---
        self.posture: str = "UNKNOWN"                  # Tư thế: STANDING, SITTING, LYING, BENDING, etc.
        self.action: str = "UNKNOWN"                   # Hành động: WALKING, RUNNING, FALLING, IDLE, LOITERING, etc.
        self.attributes: Dict[str, Any] = {}           # Các thuộc tính bổ sung (Ví dụ: has_helmet=True, holding_object=False)

    def update_position(self, bbox, keypoints=None, current_time: float = None):
        self.last_seen = current_time if current_time is not None else time.time()
        self.bbox = bbox
        if keypoints:
            self.keypoints = keypoints

    def update_behavior(self, posture: str, action: str, attributes: Optional[Dict[str, Any]] = None):
        """Cập nhật trạng thái tư thế và hành vi của đối tượng."""
        self.posture = posture
        self.action = action
        if attributes:
            self.attributes.update(attributes)

    # def check_roi_relationship(self, roi_id: str, is_currently_inside: bool, current_time: float):
    #     """Cập nhật trạng thái tương tác với một ROI cố định cụ thể."""
    #     was_inside = self.is_inside_roi.get(roi_id, False)
    #     self.is_inside_roi[roi_id] = is_currently_inside

    #     if is_currently_inside and not was_inside:
    #         # Sự kiện: Vừa bước vào ROI
    #         self.roi_entry_time[roi_id] = current_time
    #         self.crossing_history.append(f"ENTERED_{roi_id}")
            
    #     elif not is_currently_inside and was_inside:
    #         final_dwell = self.dwell_time.get(roi_id, 0.0)
    #         if roi_id in self.roi_entry_time:
    #             del self.roi_entry_time[roi_id]
    #         self.dwell_time[roi_id] = 0.0  # reset, không để giá trị cũ trôi nổi
    #         self.crossing_history.append(f"EXITED_{roi_id}_after_{final_dwell:.1f}s")
    #         alert_name = f"LOITERING_IN_{roi_id}"
    #         if alert_name in self.triggered_alerts:
    #             self.triggered_alerts.remove(alert_name)   # cho phép cảnh báo lại ở lượt ghé sau
    #         # self.crossing_history.append(f"EXITED_{roi_id}")

    #     # Cập nhật dwell time nếu vẫn đang ở trong ROI
    #     if is_currently_inside and roi_id in self.roi_entry_time:
    #         self.dwell_time[roi_id] = current_time - self.roi_entry_time[roi_id]

 
import time
from typing import List, Tuple, Dict, Any, Optional

class Human:
    def __init__(self, bbox: List[float]):
        self.bbox = bbox

    def check_iou_with_roi(self, roi_bbox: List[float]) -> float:
        """Tính toán Intersection over Union (IoU) giữa bounding box của con người và ROI."""
        human_x1, human_y1, human_x2, human_y2 = self.bbox
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox

        # Tính toán diện tích giao nhau
        inter_x1 = max(human_x1, roi_x1)
        inter_y1 = max(human_y1, roi_y1)
        inter_x2 = min(human_x2, roi_x2)
        inter_y2 = min(human_y2, roi_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Tính toán diện tích của cả hai bounding box
        human_area = (human_x2 - human_x1) * (human_y2 - human_y1)
        roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)

        # Tính toán IoU
        union_area = human_area + roi_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou
class HumanBehavior(Human):
    def __init__(self, track_id: int, current_time: float = None):
        self.track_id = track_id                                                    # ID duy nhất từ thuật toán Tracking (ByteTrack/SORT)            
        self.first_seen = current_time if current_time is not None else time.time() # Thời điểm xuất hiện lần đầu (timestamp)                                                            
        self.last_seen = self.first_seen                                            # Thời điểm cập nhật khung hình gần nhất                       
        
        # --- 2. Thông tin không gian & Vị trí (Spatial Attributes) ---
        self.bbox: List[float] = []                    # Tọa độ bounding box hiện tại: [xmin, ymin, xmax, ymax]
        self.foot_point: Tuple[float, float] = (0.0, 0.0) # Điểm neo chân (x_center, y_max) dùng để check ROI
        self.keypoints: List[List[float]] = []         # Tọa độ các khớp xương (nếu dùng Pose Estimation)
        
        # --- 3. Trương tác với ROI cố định (ROI Interaction) ---
        self.is_inside_roi: Dict[str, bool] = {}       # Trạng thái hiện tại với từng ROI (Key: roi_id, Value: True/False)
        self.roi_entry_time: Dict[str, float] = {}     # Thời điểm bắt đầu bước vào từng ROI
        self.dwell_time: Dict[str, float] = {}         # Thời gian lưu trú (Dwell time) tính bằng giây trong từng ROI
        self.crossing_history: List[str] = []          # Lịch sử sự kiện (Ví dụ: "ENTER_ROI_1", "EXIT_ROI_1")

        # --- 4. Trạng thái hành vi (Behavioral State) ---
        self.posture: str = "UNKNOWN"                  # Tư thế: STANDING, SITTING, LYING, BENDING, etc.
        self.action: str = "UNKNOWN"                   # Hành động: WALKING, RUNNING, FALLING, IDLE, LOITERING, etc.
        self.attributes: Dict[str, Any] = {}           # Các thuộc tính bổ sung (Ví dụ: has_helmet=True, holding_object=False)
        
        # --- 5. Trạng thái cảnh báo (Alert Status) ---
        self.triggered_alerts: List[str] = []          # Danh sách các cảnh báo đã kích hoạt (Ví dụ: "ZONE_INTRUSION", "FALL_DETECTED")
        self.risk_score: float = 0.0                   # Điểm rủi ro tổng hợp (Thang đo từ 0.0 đến 1.0)

    def update_position(self, bbox, keypoints=None, current_time: float = None):
        self.last_seen = current_time if current_time is not None else time.time()
        self.bbox = bbox
        # Tính điểm neo chân (đáy bounding box)
        self.foot_point = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
        if keypoints:
            self.keypoints = keypoints

    def check_roi_relationship(self, roi_id: str, is_currently_inside: bool, current_time: float):
        """Cập nhật trạng thái tương tác với một ROI cố định cụ thể."""
        was_inside = self.is_inside_roi.get(roi_id, False)
        self.is_inside_roi[roi_id] = is_currently_inside

        if is_currently_inside and not was_inside:
            # Sự kiện: Vừa bước vào ROI
            self.roi_entry_time[roi_id] = current_time
            self.crossing_history.append(f"ENTERED_{roi_id}")
            
        elif not is_currently_inside and was_inside:
            final_dwell = self.dwell_time.get(roi_id, 0.0)
            if roi_id in self.roi_entry_time:
                del self.roi_entry_time[roi_id]
            self.dwell_time[roi_id] = 0.0  # reset, không để giá trị cũ trôi nổi
            self.crossing_history.append(f"EXITED_{roi_id}_after_{final_dwell:.1f}s")
            alert_name = f"LOITERING_IN_{roi_id}"
            if alert_name in self.triggered_alerts:
                self.triggered_alerts.remove(alert_name)   # cho phép cảnh báo lại ở lượt ghé sau
            # self.crossing_history.append(f"EXITED_{roi_id}")

        # Cập nhật dwell time nếu vẫn đang ở trong ROI
        if is_currently_inside and roi_id in self.roi_entry_time:
            self.dwell_time[roi_id] = current_time - self.roi_entry_time[roi_id]

    def update_behavior(self, posture: str, action: str, attributes: Optional[Dict[str, Any]] = None):
        """Cập nhật trạng thái tư thế và hành vi của đối tượng."""
        self.posture = posture
        self.action = action
        if attributes:
            self.attributes.update(attributes)

    def evaluate_rules(self, business_rules: Dict[str, Any]) -> List[str]:
        """
        Đánh giá logic nghiệp vụ dựa trên trạng thái hiện tại và ROI.
        Trả về danh sách các cảnh báo mới cần phát ra.
        """
        new_alerts = []
        
        # Ví dụ logic: Kiểm tra lảng vảng (Loitering) trong ROI cấm
        for roi_id, inside in self.is_inside_roi.items():
            if inside:
                max_allowed_time = business_rules.get(f"max_time_{roi_id}", 30.0)
                if self.dwell_time.get(roi_id, 0) > max_allowed_time:
                    alert_name = f"LOITERING_IN_{roi_id}"
                    if alert_name not in self.triggered_alerts:
                        self.triggered_alerts.append(alert_name)
                        new_alerts.append(alert_name)
                        
        # Ví dụ logic: Phát hiện té ngã
        if self.action == "FALLING" or self.posture == "LYING":
            if "FALL_DETECTED" not in self.triggered_alerts:
                self.triggered_alerts.append("FALL_DETECTED")
                new_alerts.append("FALL_DETECTED")
                
        return new_alerts
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class ROI:
    def __init__(self, roi_id: str, name: str, polygon_points: List[Tuple[int, int]], roi_type: str = "RESTRICTED"):
        # --- 1. Định danh & Thuộc tính cơ bản ---
        self.roi_id: str = roi_id                  # Mã định danh duy nhất (Ví dụ: "ROI_01")
        self.name: str = name                      # Tên hiển thị (Ví dụ: "Khu vực cấm vào")
        self.roi_type: str = roi_type              # Loại ROI: RESTRICTED, CHECKOUT, QUEUE, DWELL_ZONE, etc.
        
        # --- 2. Hình học không gian (Geometry) ---
        # Danh sách các điểm đa giác [(x1, y1), (x2, y2), ...]
        self.polygon: List[Tuple[int, int]] = polygon_points
        self.np_polygon: np.ndarray = np.array(polygon_points, dtype=np.int32)
        
        # Tính toán bounding box bao ngoài ROI (dùng để lọc nhanh sơ bộ - pre-filtering)
        if polygon_points:
            xs = [p[0] for p in polygon_points]
            ys = [p[1] for p in polygon_points]
            self.bbox: Tuple[int, int, int, int] = (min(xs), min(ys), max(xs), max(ys)) # xmin, ymin, xmax, ymax
        else:
            self.bbox = (0, 0, 0, 0)

        # --- 3. Cấu hình quy tắc nghiệp vụ riêng cho ROI này ---
        self.rules: Dict[str, Any] = {
            "max_dwell_time": None,    # Thời gian tối đa cho phép ở lại (giây)
            "enabled": True,           # Trạng thái bật/tắt ROI này
            "alert_level": "LOW"    # Mức độ cảnh báo: LOW, MEDIUM, HIGH, CRITICAL
        }
        
        # --- 4. Trạng thái thời gian thực (Runtime Tracking) ---
        self.current_occupants: set = set()   # Tập hợp các track_id đang có mặt bên trong ROI này

    def update_rules(self, new_rules: Dict[str, Any]):
        """Cập nhật các quy tắc nghiệp vụ cho ROI."""
        self.rules.update(new_rules)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Kiểm tra một điểm (ví dụ: điểm chân người) có nằm trong đa giác ROI hay không.
        Sử dụng thuật toán Ray Casting thông qua OpenCV (cv2.pointPolygonTest).
        """
        if not self.rules.get("enabled", True) or not self.polygon:
            return False
            
        x, y = point
        # Nhanh chóng loại trừ bằng bounding box trước khi tính toán phức tạp
        xmin, ymin, xmax, ymax = self.bbox
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
            
        # Sử dụng cv2.pointPolygonTest (đã tối ưu C++)
        # Trả về: > 0 (bên trong), == 0 (trên biên), < 0 (bên ngoài)
        import cv2
        dist = cv2.pointPolygonTest(self.np_polygon, (float(x), float(y)), False)
        return dist >= 0

    def check_human_interaction(self, human_obj: Any, current_time: float) -> bool:
        if not self.rules.get("enabled", True):
            return False

        foot_point = human_obj.foot_point
        is_inside = self.contains_point(foot_point)

        if is_inside:
            self.current_occupants.add(human_obj.track_id)
        else:
            self.current_occupants.discard(human_obj.track_id)

        human_obj.check_roi_relationship(self.roi_id, is_inside, current_time)
        return is_inside

    def get_occupant_count(self) -> int:
        """Trả về số lượng người hiện đang có mặt trong ROI."""
        return len(self.current_occupants)
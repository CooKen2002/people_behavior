import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from shapely.geometry import Point, Polygon
class ROI:
    def __init__(self, id: str, state: Optional[Dict[str, Any]] = None):
        self.id = id
        self.state = state if state is not None else {}

    def get_roi_type(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    
class PolygonRoi(ROI):
    def __init__(self, id: str, polygon: List[Any], state: Optional[Dict[str, Any]] = None):
        # Truyền id và state lên lớp cha ROI
        super().__init__(id=id, state=state)
        self.polygon = polygon

    @staticmethod
    def check_point_in_polygon(self, point):
        """Kiểm tra một điểm (x, y) có nằm trong đa giác ROI hay không."""
        poly = Polygon(self.polygon)
        pt = Point(point)
        return poly.contains(pt)

class CircleRoi(ROI):
    def __init__(self, center: Tuple[float, float], radius: float):
        super().__init__()
        self.center = center
        self.radius = radius

    def get_center(self) -> Tuple[float, float]:
        return self.center

    def get_radius(self) -> float:
        return self.radius
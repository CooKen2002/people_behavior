import numpy as np
from typing import List, Tuple, Dict, Any, Optional
class ROI:
    def __init__(self, state: Optional[Dict[str, Any]] = None):
        self.state = state if state is not None else {}

    def get_roi_type(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    
class PolygonRoi(ROI):
    def __init__(self, polygon: List[float]):
        super().__init__()
        self.polygon = polygon

    def get_polygon(self) -> List[Tuple[float, float]]:
        return self.polygon

class CircleRoi(ROI):
    def __init__(self, center: Tuple[float, float], radius: float):
        super().__init__()
        self.center = center
        self.radius = radius

    def get_center(self) -> Tuple[float, float]:
        return self.center

    def get_radius(self) -> float:
        return self.radius
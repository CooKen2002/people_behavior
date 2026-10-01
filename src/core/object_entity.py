

from typing import List

class Object:
    def __init__(self, bbox: List[float]):
        self.bbox = bbox

    def check_with_human(self, human_bbox: List[float]) -> bool:
        """Kiểm tra xem đối tượng có nằm trong bounding box của con người hay không."""
        obj_x1, obj_y1, obj_x2, obj_y2 = self.bbox
        human_x1, human_y1, human_x2, human_y2 = human_bbox

        # Kiểm tra sự chồng lấp giữa hai bounding box
        overlap_x = max(0, min(obj_x2, human_x2) - max(obj_x1, human_x1))
        overlap_y = max(0, min(obj_y2, human_y2) - max(obj_y1, human_y1))

        return overlap_x > 0 and overlap_y > 0
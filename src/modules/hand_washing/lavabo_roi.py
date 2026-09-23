from ...core.roi import ROI

class HandRoi(ROI):
    def __init__(self, bbox, id):
        super().__init__(bbox, id)

    def is_washing(self, right_wrist, left_wrist) -> bool:
        top = self.bbox['x']
        left = self.bbox['y']
        width = self.bbox['w']
        height = self.bbox['h']

        if (top < right_wrist.x < top + width and left < right_wrist.y < left + height) and (top < left_wrist.x < top + width and left < left_wrist.y < left + height):
            return True
        else:
            return False


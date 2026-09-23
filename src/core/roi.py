class ROI():
    def __init__(self, bbox, id):
        self.bbox = bbox
        self.id = id

    def bbox_xywh(self):
        return self.bbox

    def load_json(self):
        pass

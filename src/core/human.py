class Human():
    def __init__(self, bbox, keypoints, state: str = None):
        self.bbox = bbox
        self.keypoints = keypoints
        self.state = state

    pass
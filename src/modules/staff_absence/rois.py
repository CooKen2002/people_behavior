import cv2
import os
import onnxruntime as ort
from ...core.roi import ROI
from ...core.frame import Frame
from utils.json_utils import *
from utils.yaml_utils import *


class WorkRois(ROI):
    def __init__(self, bbox, region_id: str, employee_id: str, region_state: str, employee_state: str):
        super().__init__(bbox)
        self.region_id = region_id
        self.employee_id = employee_id
        self.region_state = region_state
        self.employee_state = employee_state

def get_rois(frame: Frame, config):
    result_json = []

    session = ort.InferenceSession(config['mode_path'], providers=config['providers'])
    output = frame.infer(session)
    indices, boxes, confidences, class_ids = frame.parse_object_detection(output, config['conf_threshold'], config['nms_threshold'], allow_classes=[0])

    if len(indices) == 0:
        return False

    for i in indices.flatten():
        left, top, width, height = boxes[i]
        bbox_xywh = [left, top, width, height]
        region_id = ""
        employee_id=""
        region_state=""
        employee_state=""

        result_json.append(WorkRois(bbox_xywh, region_id, employee_id, region_state, employee_state))

    return result_json
if __name__ == "__main__":
    base_config = load_yaml("configs/base.yaml")
    config = load_yaml("configs/staff_absence.yaml")

    cap = cv2.VideoCapture(config['short_video_path'])

    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {config['short_video_path']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    num_frames = int(fps * config['calibration_secs'])

    result_json = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if result_json is None:
            result_json.append(get_rois(frame, base_config))     
        else:
            frame.get
import cv2
import json
import numpy as np
import onnxruntime as ort

from ...core.frame import *
from ...core.video_loader import *
from ...core.roi import *

from ...utils.json_utils import *
from ...utils.yaml_utils import *

from ...core.roi import ROI

class HandRoi(ROI):
    def __init__(self, bbox, confidence):
        super().__init__(bbox)
        self.confidence = confidence

    def is_washing(self, right_wrist, left_wrist) -> bool:
        # bbox chuẩn: [left, top, width, height] tương ứng [x, y, w, h]
        x, y, width, height = self.bbox
        
        # Hỗ trợ lấy tọa độ từ list/tuple [x, y, conf] hoặc object
        def get_xy(wrist):
            if hasattr(wrist, 'x') and hasattr(wrist, 'y'):
                return wrist.x, wrist.y
            return wrist[0], wrist[1]
            
        rx, ry = get_xy(right_wrist)
        lx, ly = get_xy(left_wrist)
        
        # Kiểm tra xem cả cổ tay phải và cổ tay trái có nằm trong vùng ROI (bồn rửa) không
        right_in = (x <= rx <= x + width) and (y <= ry <= y + height)
        left_in = (x <= lx <= x + width) and (y <= ly <= y + height)
        
        return right_in and left_in

def get_handroi(original_frame: Frame):
    rois = []
    allow_classes = [71] # ID của bồn rửa (sink) trong COCO dataset

    base_config = load_yaml('configs/base.yaml') 
    config = load_yaml('configs/hand_washing.yaml')
    model_path = config['obj_model_path']
    conf_threshold = base_config['confidence_threshold']
    nms_threshold = base_config['iou_threshold']

    session = ort.InferenceSession(model_path, providers=base_config['providers'])
    output = original_frame.infer(session, mode="resize")
    indices, boxes, confidences, _ = original_frame.parse_object_detection(output, conf_threshold, nms_threshold, allow_classes, mode="resize")

    if len(indices) > 0:
        for i in indices.flatten():
            rois.append(HandRoi(boxes[i], confidences[i]))

    return rois

def select_roi_points_from_image(image_input, window_name="Select ROI - Click 2 points (Top-Left, Bottom-Right)"):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    elif hasattr(image_input, 'frame'): # Trường hợp truyền vào đối tượng Frame
        img = image_input.frame.copy()
    else:
        print("Error: Định dạng ảnh không hợp lệ!")
        return []

    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = img.copy()
        for p in points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
            
        if len(points) == 2:
            cv2.rectangle(display, points[0], points[1], (255, 0, 255), 2)

        cv2.putText(display, "Click 2 diem (Top-Left, Bottom-Right) - q: Huy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            points = []
            break
        if len(points) == 2:
            cv2.waitKey(400)
            break

    cv2.destroyWindow(window_name)

    if len(points) < 2:
        return []

    (x1, y1), (x2, y2) = points
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    return [HandRoi([left, top, width, height], 1.0)]
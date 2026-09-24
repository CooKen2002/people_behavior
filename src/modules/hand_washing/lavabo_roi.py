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
        top, left, width, height = self.bbox
        if (top < right_wrist.x < top + width and left < right_wrist.y < left + height) and (top < left_wrist.x < top + width and left < left_wrist.y < left + height):
            return True
        else:
            return False

def get_handroi(original_frame):
    rois = []
    allow_classes = [71] # sink id

    base_config = load_yaml('configs/base.yaml') 
    config = load_yaml('configs/hand_washing.yaml')
    model_path = config['obj_model_path']
    conf_threshold = config['conf_threshold']
    nms_threshold = config['nms_threshold']
    input_shape = tuple(config['input_shape'])

    frame = Frame(cv2.imread(original_frame), input_shape)
    session = ort.InferenceSession(model_path, providers=base_config['providers'])

    indices, boxes, confidences, _ = frame.obj_inference(conf_threshold, nms_threshold, allow_classes, session, mode="resize")

    if len(indices) > 0:
        for i in indices.flatten():
            rois.append(HandRoi(boxes[i], confidences[i]))

    return rois

def select_roi_points_from_image(image_input, window_name="Select ROI - Click 2 points (Top-Left, Bottom-Right)"):
    """
    Chọn vùng ROI thủ công từ một ảnh tĩnh (đường dẫn file hoặc numpy array).
    Bấm 'q' hoặc ESC để hủy, tự động hoàn thành khi chọn đủ 2 điểm.
    """
    # Đọc ảnh nếu đầu vào là đường dẫn (str), ngược lại coi như đã là numpy.ndarray
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            print(f"Error: Không thể đọc ảnh từ đường dẫn: {image_input}")
            return None
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        print("Error: Định dạng ảnh không hợp lệ!")
        return None

    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = img.copy()

        # Vẽ các điểm đã click
        for p in points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
            
        # Nếu chọn đủ 2 điểm, vẽ hình chữ nhật tạm thời
        if len(points) == 2:
            cv2.rectangle(display, points[0], points[1], (255, 0, 255), 2)

        cv2.putText(
            display,
            "Click 2 diem (Top-Left, Bottom-Right) - q: Huy",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        # Thoát nếu bấm 'q' hoặc phím ESC (27)
        if key == ord("q") or key == 27:
            points = []
            break

        # Đã chọn đủ 2 điểm thì dừng vòng lặp sau một khoảng dừng ngắn
        if len(points) == 2:
            cv2.waitKey(400)
            break

    cv2.destroyWindow(window_name)

    if len(points) < 2:
        return None

    # Tính toán tọa độ chuẩn (left, top, width, height)
    (x1, y1), (x2, y2) = points
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    rois = []
    rois.append(HandRoi([left, top, width, height],1))
    return rois
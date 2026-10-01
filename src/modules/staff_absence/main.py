import os
import cv2
import json
import numpy as np
import onnxruntime as ort

from ...core.frame import *
from ...core.human import *
from ...core.roi import *
from ...utils.yaml_utils import load_yaml
from ...utils.json_utils import load_json, save_json  # Đảm bảo đúng hàm

from .draw_rois import select_multiple_rois

def get_rois(mode, image_path, output_path):
    # Sửa lại điều kiện kiểm tra mode cho chính xác
    if mode == "manual" or mode is None:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        rois = select_multiple_rois(image, window_name="Select Multiple ROIs")
        
        # Sửa lại đúng thứ tự tham số: (đường dẫn, dữ liệu)
        save_json(f'{output_path}/rois.json', rois)
        
        return rois
    else:
        pass

if __name__ == "__main__":
    base_config = load_yaml('configs/base.yaml')
    config = load_yaml('configs/staff_absence.yaml')

    session = ort.InferenceSession(config['pose_model_path'], providers=base_config['providers'])

    
    if os.path.exists(f"{base_config['annotated_path']}/rois.json"):
        rois = load_json(f"{base_config['annotated_path']}/rois.json")
    else:
        rois = get_rois(config['get_rois'], f"{base_config['raw_path']}/1809_frame.jpg", base_config['annotated_path'])

    cap = cv2.VideoCapture(base_config['raw_path'] + "/1809.mp4")

    while cap.isOpened():
        ret, vid_frame = cap.read()

        if not ret:
            break

        frame = Frame(vid_frame)
        output = frame.infer(session, mode="resize")
        frame.parse_pose_estimation(
            output=output,
            conf_threshold=base_config['conf_threshold'],
            nms_threshold=base_config['nms_threshold'],
            allow_classes=[0],  # Chỉ cho phép class "person"
            mode="resize"
        )

        

        # Xử lý frame ở đây
        # Ví dụ: hiển thị frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
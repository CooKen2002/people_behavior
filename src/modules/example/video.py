import cv2
import json
import numpy as np
import onnxruntime as ort

from ...core.frame import *
from ...core.video_loader import *
from ...core.roi import *

from ...utils.json_utils import *
from ...utils.yaml_utils import *

base_config = load_yaml("configs/base.yaml")
config = load_yaml("configs/example.yaml")

session = ort.InferenceSession(config["model_path"], providers=base_config["providers"])

capCam = cv2.VideoCapture(config["video_path"])

if not capCam.isOpened():
    print("Error: Can't open cam or video.")
    exit()

window_name = "Example"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Lấy thông số của video gốc để cấu hình ghi file video đầu ra
fps = capCam.get(cv2.CAP_PROP_FPS)
frame_width = int(capCam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capCam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while capCam.isOpened():
    ret, cam_frame = capCam.read()
    if not ret:
        print("CLOSE")
        break

    frame = Frame(cam_frame, base_config["target_size"])
    output = frame.infer(session)
    indices, boxes, confidences, class_ids = frame.parse_object_detection(
        output,
        base_config["conf_threshold"],
        base_config["nms_threshold"],
        allow_classes=[0],
        mode="resize",
    )

    for i in indices.flatten():
        top, left, width, height = boxes[i]
        cv2.rectangle(
            cam_frame,
            (top, left),
            (top + width, left + height),
            base_config["red"],
            base_config["thickness"],
        )
        text = f"{class_ids[i]}.conf: {confidences[i]:2f}"
        cv2.putText(
            cam_frame,
            text,
            org=(top, left),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=base_config["text_scale"],
            color=base_config["green"],
            thickness=base_config["thickness"],
        )
    # Hiển thị frame đã vẽ (dùng original.frame thay vì frame)
    cv2.imshow(window_name, cam_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
capCam.release()
cv2.destroyAllWindows()

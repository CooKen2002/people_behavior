import cv2
import json
import numpy as np
import onnxruntime as ort

from ...core.frame import *
from ...core.video_loader import *
from ...core.roi import *

from ...utils.json_utils import *
from ...utils.yaml_utils import *


base_config = load_yaml('configs/base.yaml') 
config = load_yaml('configs/example.yaml')

model_path = config['model_path']
video_path = config['video_path']

conf_threshold = config['conf_threshold']
nms_threshold = config['nms_threshold']
input_shape = tuple(config['input_shape'])

output_path = config['output_path']
output_file = f'{output_path}/example_result.mp4'

session = ort.InferenceSession(model_path, providers=base_config['providers'])
input_name = session.get_inputs()[0].name

CLASSES = base_config['coco_object_classes']

capCam = cv2.VideoCapture(video_path)
if not capCam.isOpened():
    print("Error: Can't open cam or video.")
    exit()

# Lấy thông số của video gốc để cấu hình ghi file video đầu ra
fps = capCam.get(cv2.CAP_PROP_FPS)
frame_width = int(capCam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capCam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Khởi tạo VideoWriter để lưu video kết quả
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
cv2.namedWindow("Example", cv2.WINDOW_NORMAL)

while capCam.isOpened():
    ret, frame = capCam.read()
    if not ret:
        print("CLOSE")
        break

    # Đưa frame vào class Frame để xử lý
    original = Frame(frame)
    preprocess_frame = original.preprocess(input_shape)

    outputs = session.run(None, {input_name: preprocess_frame})
    output = outputs[0][0]  # Shape: [84, 8400]
    output = output.T       # Transpose to [8400, 84]

    boxes = []
    confidences = []
    class_ids = []

    x_factor = original.width / input_shape[0]
    y_factor = original.heigth / input_shape[1]

    for row in output:
        classes_scores = row[4:]
        max_score = np.amax(classes_scores)
        
        if max_score >= conf_threshold:
            class_id = np.argmax(classes_scores)
            
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            left = round((cx - w / 2) * x_factor)
            top = round((cy - h / 2) * y_factor)
            width = round(w * x_factor)
            height = round(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            left, top, width, height = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            class_name = CLASSES[class_id]

            # Vẽ bounding box lên original.frame
            cv2.rectangle(original.frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            
            label = f"Class {class_id}_{class_name}: {confidence:.2f}"
            cv2.putText(original.frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, base_config['visualization']['text_scale'], (0, 255, 0), 2)
            
    # Ghi frame đã vẽ bounding box vào file video kết quả
    out.write(original.frame)

    # Hiển thị frame đã vẽ (dùng original.frame thay vì frame)
    cv2.imshow("Example", original.frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
capCam.release()
out.release()
cv2.destroyAllWindows()
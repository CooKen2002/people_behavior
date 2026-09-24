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

model_path = config["model_path"]
image_path = config["image_path"]

conf_threshold = config["confidence_threshold"]
nms_threshold = config["nms_threshold"]
input_shape = tuple(config["input_shape"])

output_path = config["output_path"]
output_file = f"{output_path}/example_result.jpg"


session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

original_image = Frame(cv2.imread(image_path))
preprocess_frame = original_image.preprocess(input_shape)

# # 3. Model Inference
# outputs = session.run(None, {input_name: input_image})
# output = outputs[0][0]  # Shape: [num_features, num_predictions] (e.g., [84, 8400])
# output = output.T        # Transpose to [num_predictions, num_features]

outputs = session.run(None, {input_name: preprocess_frame})
output = outputs[0][0]  # Shape: [num_features, num_predictions] (e.g., [84, 8400])
output = output.T  # Transpose to [num_predictions, num_features]

# 4. Parse Outputs (Boxes, Confidences, Class IDs)
boxes = []
confidences = []
class_ids = []

# Scaling ratios back to the original image dimensions
x_factor = original_image.width / input_shape[0]
y_factor = original_image.heigth / input_shape[1]

for row in output:
    classes_scores = row[4:]
    max_score = np.amax(classes_scores)

    if max_score >= conf_threshold:
        class_id = np.argmax(classes_scores)

        # YOLO formats bounding boxes as [x_center, y_center, width, height]
        cx, cy, w, h = row[0], row[1], row[2], row[3]

        # Convert to [left, top, width, height] in pixel coordinates use round instead int for closest result of bbox
        left = round((cx - w / 2) * x_factor)
        top = round((cy - h / 2) * y_factor)
        width = round(w * x_factor)
        height = round(h * y_factor)

        boxes.append([left, top, width, height])
        confidences.append(float(max_score))
        class_ids.append(class_id)

# 5. Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# 6. Draw Bounding Boxes on Original Image
if len(indices) > 0:
    for i in indices.flatten():
        left, top, width, height = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        class_name = CLASSES[class_id]

        # Draw bounding box rectangle
        cv2.rectangle(
            original_image.frame,
            (left, top),
            (left + width, top + height),
            (0, 255, 0),
            2,
        )

        # Create and draw text label
        label = f"Class {class_id}_{class_name}: {confidence:.2f}"
        cv2.putText(
            original_image.frame,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

# Save or display the resulting image
cv2.imwrite(output_file, original_image.frame)
cv2.imshow("Detections", original_image.frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

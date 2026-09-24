import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional, List, Dict, Any

class Frame:
    def __init__(self, frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None):
        self.frame = frame
        self.target_size = target_size or (640, 640)
        self.height, self.width = self.frame.shape[:2]
        self.ratio, self.dw, self.dh = 1.0, 0.0, 0.0

    def _preprocess(self, mode: str = "resize") -> np.ndarray:
        """Xử lý tiền xử lý tập trung cho mọi loại model YOLO ONNX"""
        ih, iw = self.height, self.width
        tw, th = self.target_size

        if mode == "resize":
            resized = cv2.resize(self.frame, (tw, th))
            self.ratio = tw / iw # Hoặc lưu hệ số scale x/y riêng nếu cần
            self.dw, self.dh = 0.0, 0.0
        elif mode == "letterbox":
            r = min(th / ih, tw / iw)
            nw, nh = int(round(iw * r)), int(round(ih * r))
            resized = cv2.resize(self.frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            
            dw, dh = (tw - nw) / 2, (th - nh) / 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            
            resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            self.ratio, self.dw, self.dh = r, dw, dh
        else:
            raise ValueError("Mode không hợp lệ!")

        # Chuẩn hóa chung CHW và scale [0, 1]
        tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return np.expand_dims(tensor, axis=0).astype(np.float32) / 255.0

    def infer(self, session: ort.InferenceSession, mode: str = "resize") -> np.ndarray:
        """Thực hiện chạy forward pass ONNX chung"""
        input_name = session.get_inputs()[0].name
        tensor = self._preprocess(mode)
        outputs = session.run(None, {input_name: tensor})
        return outputs[0][0].T  # Shape chuẩn sau transpose: [N, Features]

    def _map_coordinate(self, cx: float, cy: float, w: float, h: float, mode: str) -> Tuple[int, int, int, int]:
        """Ánh xạ tọa độ bounding box từ tensor về ảnh gốc"""
        if mode == "resize":
            x_factor = self.width / self.target_size[0]
            y_factor = self.height / self.target_size[1]
            left = round((cx - w / 2) * x_factor)
            top = round((cy - h / 2) * y_factor)
            width = round(w * x_factor)
            height = round(h * y_factor)
        else:
            left = round((cx - w / 2 - self.dw) / self.ratio)
            top = round((cy - h / 2 - self.dh) / self.ratio)
            width = round(w / self.ratio)
            height = round(h / self.ratio)
        return left, top, width, height

    def parse_object_detection(self, output: np.ndarray, conf_threshold: float, nms_threshold: float, allow_classes: List[int], mode: str):
        boxes, confidences, class_ids = [], [], []

        for row in output:
            classes_scores = row[4:]
            max_score = np.amax(classes_scores)
            if max_score >= conf_threshold:
                class_id = int(np.argmax(classes_scores))
                if allow_classes and class_id not in allow_classes:
                    continue
                
                cx, cy, w, h = row[:4]
                left, top, width, height = self._map_coordinate(cx, cy, w, h, mode)
                
                boxes.append([left, top, width, height])
                confidences.append(float(max_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        return indices, boxes, confidences, class_ids

    def parse_pose_estimation(self, output: np.ndarray, conf_threshold: float, nms_threshold: float, mode: str):
        boxes, confidences, keypoints_list = [], [], []

        for row in output:
            box_conf = row[4]
            if box_conf >= conf_threshold:
                cx, cy, w, h = row[:4]
                left, top, width, height = self._map_coordinate(cx, cy, w, h, mode)

                # Giải mã 17 keypoints (mỗi keypoint gồm x, y, conf)
                kpts_raw = row[5:]
                kpts = []
                for i in range(0, len(kpts_raw), 3):
                    kx, ky, kconf = kpts_raw[i], kpts_raw[i+1], kpts_raw[i+2]
                    # Map tọa độ keypoint tương tự bbox
                    if mode == "resize":
                        orig_kx = round(kx * (self.width / self.target_size[0]))
                        orig_ky = round(ky * (self.height / self.target_size[1]))
                    else:
                        orig_kx = round((kx - self.dw) / self.ratio)
                        orig_ky = round((ky - self.dh) / self.ratio)
                    kpts.append([orig_kx, orig_ky, float(kconf)])

                boxes.append([left, top, width, height])
                confidences.append(float(box_conf))
                keypoints_list.append(kpts)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        return indices, boxes, confidences, keypoints_list
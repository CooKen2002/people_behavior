import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional


class Frame:
    def __init__(
        self, frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None
    ):
        self.frame = frame
        self.target_size = target_size or (
            640,
            640,
        )  # Giá trị mặc định nếu không truyền
        self.height, self.width = self.frame.shape[:2]

    def _preprocess_resize(self):
        """Mode 1: Resize trực tiếp (Nhanh, có thể méo hình nếu khác tỉ lệ)"""
        preprocess_frame = cv2.resize(self.frame, self.target_size)
        preprocess_frame = cv2.cvtColor(preprocess_frame, cv2.COLOR_BGR2RGB)
        preprocess_frame = preprocess_frame.transpose(2, 0, 1)  # HWC to CHW
        preprocess_frame = np.expand_dims(preprocess_frame, axis=0).astype(np.float32)
        preprocess_frame /= 255.0
        return preprocess_frame

    def _preprocess_letterbox(self):
        """Mode 2: Letterbox (Chính xác cao, giữ nguyên tỉ lệ khung hình, thêm viền xám)"""
        ih, iw = self.height, self.width
        tw, th = self.target_size  # (width, height)
        # Các biến lưu thông số letterbox (nếu dùng mode letterbox)
        r = min(th / ih, tw / iw)
        nw, nh = int(round(iw * r)), int(round(ih * r))

        resized = cv2.resize(self.frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        dw, dh = tw - nw, th - nh
        dw /= 2
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padded_img = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Chuẩn hóa tensor cho ONNX
        preprocess_frame = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        preprocess_frame = preprocess_frame.transpose(2, 0, 1)
        preprocess_frame = np.expand_dims(preprocess_frame, axis=0).astype(np.float32)
        preprocess_frame /= 255.0

        return preprocess_frame, r, dw, dh

    def obj_inference(
        self,
        conf_threshold: float,
        nms_threshold: float,
        allow_classes: tuple,
        session: ort.InferenceSession,
        mode: str = "resize",
    ):
        input_name = session.get_inputs()[0].name

        # 1. Chọn phương thức preprocess dựa vào mode truyền vào
        if mode == "resize":
            preprocess_frame = self._preprocess_resize()
        elif mode == "letterbox":
            preprocess_frame, r, dw, dh = self._preprocess_letterbox()
        else:
            raise ValueError("Mode không hợp lệ! Hãy chọn 'resize' hoặc 'letterbox'.")

        # 2. Chạy Inference ONNX
        outputs = session.run(None, {input_name: preprocess_frame})
        output = outputs[0][0]  # Shape: [84, 8400]
        output = output.T  # Transpose thành [8400, 84]

        boxes = []
        confidences = []
        class_ids = []

        # 3. Tính toán hệ số scale ngược lại ảnh gốc tùy theo mode
        if mode == "resize":
            x_factor = self.width / self.target_size[0]
            y_factor = self.height / self.target_size[1]
        else:  # letterbox
            # Đối với letterbox, cần trừ đi phần padding (dw, dh) và chia cho hệ số scale (ratio)
            pass

        for row in output:
            classes_scores = row[4:]
            max_score = np.amax(classes_scores)

            if max_score >= conf_threshold:
                class_id = int(np.argmax(classes_scores))
                if allow_classes is not None and class_id not in allow_classes:
                    continue

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                if mode == "resize":
                    left = round((cx - w / 2) * x_factor)
                    top = round((cy - h / 2) * y_factor)
                    width = round(w * x_factor)
                    height = round(h * y_factor)
                else:  # letterbox mapping chuẩn xác
                    # Đưa tọa độ từ khung hình letterbox về lại khung hình gốc
                    left = round((cx - w / 2 - dw) / r)
                    top = round((cy - h / 2 - dh) / r)
                    width = round(w / r)
                    height = round(h / r)

                boxes.append([left, top, width, height])
                confidences.append(float(max_score))
                class_ids.append(class_id)

        # 4. Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        return indices, boxes, confidences, class_ids

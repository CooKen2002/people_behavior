import cv2
import json
import numpy as np
import onnxruntime as ort

from ...core.frame import Frame
from ...core.roi import ROI
from ...utils.json_utils import *
from ...utils.yaml_utils import *
from work_region import WorkRegion

# Ngưỡng "biến mất hẳn khỏi khung hình" trước khi region reset về unassigned.
# Vá bug: bản gốc không có đường nào đưa "away" quay lại "unassigned".
# Nên đặt >= TRACK_MAX_DISAPPEARED (xem giải thích trong work_region.py).

base_config = load_yaml('configs/base.yaml')
config = load_yaml('configs/staff_absence.yaml')


# ============== TRACKER ==============
# Giữ nguyên logic gốc: ONNX Runtime không có tracker tích hợp như
# Ultralytics, nên tự cài một tracker theo khoảng cách centroid giữa
# các frame liên tiếp. Giữ luôn payload (keypoints) đi kèm mỗi track
# để không phải tra cứu lại theo chỉ số sau khi ID bị xáo trộn.
class SimpleTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 1
        self.objects = {}
        self.boxes = {}
        self.payloads = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, box, payload):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.boxes[oid] = box
        self.payloads[oid] = payload
        self.disappeared[oid] = 0
        self.next_object_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.boxes[oid]
        del self.payloads[oid]
        del self.disappeared[oid]

    def update(self, rects, payloads):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {}

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2), int(y + h / 2))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i], payloads[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis, :] - input_centroids[np.newaxis, :, :], axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                oid = object_ids[row]
                self.objects[oid] = input_centroids[col]
                self.boxes[oid] = rects[col]
                self.payloads[oid] = payloads[col]
                self.disappeared[oid] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            for row in unused_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col], payloads[col])

        return {oid: (self.boxes[oid], self.payloads[oid]) for oid in self.objects.keys()}


def run_pose_inference(frame_obj: Frame, session, conf_threshold: float, iou_threshold: float):
    """
    Thay cho detect_pose_onnx() cũ — dùng Frame (frame.py) làm lớp
    preprocessing/postprocessing dùng chung, không tự viết lại resize/
    letterbox/map-coordinate riêng trong main.py nữa.

    Trả về (boxes, scores, keypoints_list) đã lọc theo NMS, cùng format
    với hàm cũ để phần code xử lý per-person phía dưới không cần đổi gì.
    """
    output = frame_obj.infer(session, mode="resize")
    indices, boxes, confidences, keypoints_list = frame_obj.parse_pose_estimation(
        output, conf_threshold, iou_threshold, mode="resize"
    )

    # cv2.dnn.NMSBoxes trả về () (tuple rỗng, không có .flatten()) khi
    # không có box nào, và ndarray phẳng khi có box — Frame không tự xử
    # lý khác biệt này nên phải làm ở đây trước khi index vào list.
    if len(indices) == 0:
        return [], [], []

    idx_flat = np.array(indices).flatten()
    final_boxes = [boxes[i] for i in idx_flat]
    final_scores = [confidences[i] for i in idx_flat]
    final_keypoints = [keypoints_list[i] for i in idx_flat]
    return final_boxes, final_scores, final_keypoints


# ============== MAIN ==============

rois_raw = load_json(config['rois_json_path'])

regions = [
    WorkRegion.from_dict(
        r,
        assign_threshold=config['assign_threshold'],
        away_threshold=config['away_threshold'],
        missing_threshold=config['missing_threshold'],
    )
    for r in rois_raw
]

session = ort.InferenceSession(config['pose_model_path'], providers=base_config['providers'])

tracker = SimpleTracker(max_disappeared=config['track_max_disappeared'], max_distance=config['max_distance'])

cap = cv2.VideoCapture(config['video_path'])

if not cap.isOpened():
    print("Error: Không thể mở video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_obj = Frame(frame, base_config['target_size'])
    boxes, scores, keypoints_list = run_pose_inference(
        frame_obj, session, base_config['conf_threshold'], base_config['nms_threshold']
    )

    tracked = tracker.update(boxes, keypoints_list)

    detected_people = []

    for track_id, (box, person_kpts) in tracked.items():
        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h

        hip_x, hip_y = None, None
        left_hip = person_kpts[11]
        right_hip = person_kpts[12]

        if left_hip[2] > config['kpt_conf_threshold'] and right_hip[2] > config['kpt_conf_threshold']:
            hip_x = int((left_hip[0] + right_hip[0]) / 2)
            hip_y = int((left_hip[1] + right_hip[1]) / 2)
        elif left_hip[2] > config['kpt_conf_threshold']:
            hip_x, hip_y = int(left_hip[0]), int(left_hip[1])
        elif right_hip[2] > config['kpt_conf_threshold']:
            hip_x, hip_y = int(right_hip[0]), int(right_hip[1])

        detected_people.append({
            "track_id": str(track_id),
            "box": (x1, y1, x2, y2),
            "hip": (hip_x, hip_y),
            "keypoints": person_kpts,
        })

        valid_kpts = {}
        for idx in config['target_indices']:
            kx, ky, kconf = person_kpts[idx]
            if kconf > config['kpt_conf_threshold']:
                valid_kpts[idx] = (int(kx), int(ky))
                cv2.circle(frame, (int(kx), int(ky)), 5, (0, 0, 255), -1)

        for p1, p2 in base_config['pose_skeleton']:
            if p1 in valid_kpts and p2 in valid_kpts:
                cv2.line(frame, valid_kpts[p1], valid_kpts[p2], (255, 0, 0), 2)

    # --- STATE MACHINE: giờ do WorkRegion tự quản lý, main.py chỉ gọi update() ---
    for region in regions:
        region.update(detected_people)

    # --- VẼ KHUNG ROI LÊN FRAME ---
    for region in regions:
        rx, ry, rw, rh = region.roi.bbox_xywh()
        color = region.display_color()

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(
            frame, region.label(), (rx, max(20, ry - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )

    cv2.imshow("Smart Workspace Management - Anti ID Switch (ONNX)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import onnxruntime as ort

from ...core.frame import Frame
from ...utils.json_utils import save_json
from ...utils.yaml_utils import load_yaml

# ============== CẤU HÌNH ==============
base_config = load_yaml('configs/base.yaml')
config = load_yaml('configs/staff_absence.yaml')


# Có thể chuyển EMPLOYEE_ASSIGNMENTS vào 1 file config hoặc database sau
EMPLOYEE_ASSIGNMENTS = {}


# ============== WRAPPER AI ==============
def run_object_inference(frame_obj: Frame, session, conf_threshold: float, iou_threshold: float):
    """
    Sử dụng Frame để chạy AI và lọc kết quả (thay cho detect_persons_in_frame cũ).
    Giả định class Frame có hỗ trợ parse_object_detection, hoặc bạn tự wrap lại như sau:
    """
    output = frame_obj.infer(session, mode="resize")
    
    # LƯU Ý: Giả định Frame có hàm này tương tự parse_pose_estimation. 
    # Nếu chưa có, bạn cần implement trong ...core.frame
    indices, boxes, confidences, class_ids = frame_obj.parse_object_detection(
        output, conf_threshold, iou_threshold, allow_classes=[0], mode="resize"
    )

    if len(indices) == 0:
        return [], []

    idx_flat = np.array(indices).flatten()
    
    final_boxes = []
    final_scores = []
    
    for i in idx_flat:
        # Chỉ lấy class_id == 0 (Person)
        if class_ids[i] == 0:
            final_boxes.append(boxes[i])
            final_scores.append(confidences[i])
            
    return final_boxes, final_scores


# ============== TIỆN ÍCH IoU & CLUSTER ==============
def iou_xywh(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = (aw * ah) + (bw * bh) - inter_area
    return inter_area / union if union > 0 else 0.0


def median_box(boxes):
    arr = np.array(boxes)
    return [
        float(np.median(arr[:, 0])),
        float(np.median(arr[:, 1])),
        float(np.median(arr[:, 2])),
        float(np.median(arr[:, 3])),
    ]


def cluster_detections(all_detections, iou_threshold):
    clusters = []
    for det in all_detections:
        box, score, frame_idx = det['box'], det['score'], det['frame_idx']
        best_cluster, best_iou = None, 0.0

        for c in clusters:
            ref_box = median_box(c['boxes'])
            i = iou_xywh(box, ref_box)
            if i > iou_threshold and i > best_iou:
                best_iou = i
                best_cluster = c

        if best_cluster is not None:
            best_cluster['boxes'].append(box)
            best_cluster['scores'].append(score)
            best_cluster['frames'].add(frame_idx)
        else:
            clusters.append({
                'boxes': [box],
                'scores': [score],
                'frames': {frame_idx}
            })
    return clusters


# ============== CORE LOGIC ==============
def run_calibration(session):
    video_path = config['video_path'] # Có thể đổi thành short_video_path nếu muốn
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Không thể mở video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    num_frames = int(fps * config.get('calibration_secs', 5))

    all_detections = []
    representative_frame = None
    frame_idx = 0

    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == num_frames // 2:
            representative_frame = frame.copy()

        # Áp dụng kiến trúc Frame mới
        frame_obj = Frame(frame, base_config['target_size'])
        boxes, scores = run_object_inference(
            frame_obj, session, base_config['conf_threshold'], base_config['nms_threshold']
        )

        for box, score in zip(boxes, scores):
            all_detections.append({'box': box, 'score': score, 'frame_idx': frame_idx})

        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        raise RuntimeError("Không đọc được frame nào.")

    clusters = cluster_detections(all_detections, config['iou_threshold_cluster'])
    rois = []
    
    for c in clusters:
        frequency = len(c['frames']) / frame_idx
        if frequency >= config['frequency_threshold']:
            rois.append({
                'box': median_box(c['boxes']),
                'frequency': frequency,
                'avg_confidence': float(np.mean(c['scores'])),
                'num_detections': len(c['boxes'])
            })

    # Sort ROI theo y (từ trên xuống dưới) và x (từ trái qua phải) để dễ quản lý
    rois.sort(key=lambda r: (round(r['box'][1] / 100), r['box'][0]))
    return rois, representative_frame, frame_idx


def pad_box(box, ratio, frame_w, frame_h):
    left, top, width, height = box
    pad_w, pad_h = width * ratio, height * ratio
    new_left = max(0, left - pad_w / 2)
    new_top = max(0, top - pad_h / 2)
    new_width = min(frame_w - new_left, width + pad_w)
    new_height = min(frame_h - new_top, height + pad_h)
    return [int(new_left), int(new_top), int(new_width), int(new_height)]


def assign_employee_ids(rois_with_bbox, representative_frame):
    preview = representative_frame.copy()
    for roi in rois_with_bbox:
        left, top, width, height = roi['bbox_xywh']
        cv2.rectangle(preview, (left, top), (left + width, top + height), (0, 200, 255), 2)
        cv2.putText(preview, roi['roi_id'], (left, max(15, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    
    cv2.imwrite('data/annotations/calibration_preview.jpg', preview)

    for roi in rois_with_bbox:
        roi_id = roi['roi_id']
        if roi_id in EMPLOYEE_ASSIGNMENTS:
            roi['assigned_employee_id'] = EMPLOYEE_ASSIGNMENTS[roi_id]
        else:
            try:
                emp_id = input(f"Nhập ID nhân viên cho {roi_id} (Enter bỏ qua): ").strip()
                roi['assigned_employee_id'] = emp_id if emp_id else None
            except EOFError:
                roi['assigned_employee_id'] = None

    return rois_with_bbox


def draw_and_save(rois, frame, output_img, output_json):
    frame_h, frame_w = frame.shape[:2]
    result_json = []

    for i, roi in enumerate(rois):
        result_json.append({
            'roi_id': f'ROI_{i+1}',
            'bbox_xywh': pad_box(roi['box'], config['frequency_threshold'], frame_w, frame_h),
            'avg_confidence': roi['avg_confidence'],
            'frequency': roi['frequency'],
            'num_detections': roi['num_detections'],
        })

    result_json = assign_employee_ids(result_json, frame)

    for roi in result_json:
        left, top, width, height = roi['bbox_xywh']
        cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 0, 0), 2)
        emp = roi['assigned_employee_id'] or "Unassigned"
        label = f"{roi['roi_id']} | {emp} | {roi['frequency']*100:.0f}%"
        cv2.putText(frame, label, (left, max(15, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_img, frame)
    save_json(output_json, result_json)
    print(f"\n[OK] Đã lưu {len(result_json)} ROI -> {output_json}")


# ============== RUNNER ==============
if __name__ == '__main__':
    # Dùng object model (YOLOv8n thuần) thay vì pose model để lấy bbox cho chuẩn
    session = ort.InferenceSession(
        config['obj_model_path'], 
        providers=base_config['providers']
    )
    
    rois, rep_frame, total_frames = run_calibration(session)

    if rep_frame is not None:
        draw_and_save(
            rois, 
            rep_frame, 
            output_img='data/annotations/work_region_visualized.jpg', 
            output_json=config['rois_json_path']
        )
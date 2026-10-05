import os
import time
import cv2
import json
import numpy as np
import onnxruntime as ort

from src.core.frame import Frame
from src.core.human import HumanBehavior
from src.core.tracker import EntityTracker
from src.utils.yaml_utils import load_yaml
from src.utils.json_utils import load_json, save_json
from .draw_rois import select_multiple_rois
from shapely.geometry import Point, Polygon


def get_rois(mode, image_path, output_path):
    """Hàm tạo và lưu danh sách ROI nếu chưa tồn tại file"""
    if mode == "manual" or mode is None:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        rois = select_multiple_rois(image, window_name="Select Multiple ROIs")
        data = [
            {
                "id": roi.id, 
                "polygon": roi.polygon, 
                "state": "absence", 
                "assigned_employee": None
            } for roi in rois
        ]
        save_json(f"{output_path}/rois.json", data)
        return data
    else:
        return []


if __name__ == "__main__":
    base_config = load_yaml("configs/base.yaml")
    config = load_yaml("configs/staff_absence.yaml")
    
    # Khởi tạo ONNX session cho mô hình Pose Estimation
    session = ort.InferenceSession(
        config["pose_model_path"], providers=base_config["providers"]
    )

    # 1. Kiểm tra hoặc khởi tạo ROIs
    rois_json_path = f"{base_config['annotated_path']}/rois.json"
    if os.path.exists(rois_json_path):
        rois = load_json(rois_json_path)
    else:
        rois = get_rois(
            config.get("get_rois", "manual"),
            f"{base_config['raw_path']}/1809_frame.jpg",
            base_config["annotated_path"],
        )

    # Đảm bảo cấu trúc mỗi roi luôn có các trường cần thiết
    for roi in rois:
        if "state" not in roi:
            roi["state"] = "absence"
        if "assigned_employee" not in roi:
            roi["assigned_employee"] = None

    # Khởi tạo video capture và tracker ở NGOÀI vòng lặp để duy trì xuyên suốt
    cap = cv2.VideoCapture(base_config["raw_path"] + "/1809.mp4")
    tracker = EntityTracker(max_missed_frames=30, iou_threshold=0.3)

    M_FRAMES = 10         # Số frame liên tiếp để chính thức gán nhân viên vào ROI
    OVERLAP_THRESHOLD = 0.25  # Ngưỡng tỷ lệ giao thoa (25% diện tích ROI)

    while cap.isOpened():
        ret, vid_frame = cap.read()
        if not ret:
            break

        frame = Frame(vid_frame)
        output = frame.infer(session, mode="resize")
        indices, boxes, confidences, keypoints_list = frame.parse_pose_estimation(
            output,
            conf_threshold=base_config["conf_threshold"],
            nms_threshold=base_config["nms_threshold"],
        )

        # Lọc danh sách detections và lưu metadata kèm theo
        current_detections = []
        det_meta = {}
        for i in indices:
            box = boxes[i]  # [left, top, width, height]
            conf = confidences[i]
            kpts = keypoints_list[i]
            
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            std_box = [x1, y1, x2, y2]
            
            current_detections.append(std_box)
            det_meta[tuple(std_box)] = {"confidence": conf, "keypoints": kpts, "raw_box": box}

        # 2. TRACKING NHÂN VIÊN THEO THỜI GIAN
        active_humans = tracker.update(detections=current_detections)

        employees = []
        for human in active_humans:
            box_tuple = tuple(human.bbox)
            meta = det_meta.get(box_tuple, {"confidence": 0.5, "keypoints": [], "raw_box": human.bbox})

            human_behavior = HumanBehavior(
                track_id=str(human.track_id),
                confidence=meta["confidence"],
                keypoints=meta["keypoints"],
                current_time=time.time(),
                bbox=human.bbox,
            )
            human_behavior.hits = human.hits
            human_behavior.time_since_update = human.time_since_update
            employees.append(human_behavior)

        # 3. LOGIC GÁN NHÂN VIÊN VÀO ROIS DỰA TRÊN TỶ LỆ GIAO THOA (CHỈ CHẠY DUY NHẤT 1 LẦN)
        assigned_emp_ids_this_frame = set()

        for roi in rois:
            roi_id = roi["id"]
            polygon_pts = roi["polygon"]
            poly_obj = Polygon(polygon_pts)
            roi_area = poly_obj.area

            current_assigned_emp = roi.get("assigned_employee")
            matched_emp = None

            # BƯỚC A: Kiểm tra nhân viên ĐÃ TỪNG ĐƯỢC GÁN xem còn giao thoa đủ lớn với ROI không
            if current_assigned_emp is not None:
                for emp in employees:
                    if emp.track_id == current_assigned_emp:
                        ex1, ey1, ex2, ey2 = emp.bbox
                        emp_box_poly = Polygon([(ex1, ey1), (ex2, ey1), (ex2, ey2), (ex1, ey2)])
                        
                        if poly_obj.intersects(emp_box_poly):
                            inter_area = poly_obj.intersection(emp_box_poly).area
                            if inter_area > 0:
                                matched_emp = emp
                        break

            # BƯỚC B: Nếu chưa có ai hoặc nhân viên cũ đã rời đi, tìm nhân viên mới có độ phủ tốt nhất
            if matched_emp is None:
                best_new_emp = None
                max_overlap_ratio = 0.0

                for emp in employees:
                    if emp.track_id in assigned_emp_ids_this_frame:
                        continue  # Nhân viên này đã bị chiếm ở ROI khác
                    
                    # Kiểm tra xem nhân viên này có đang được gán cố định ở ROI khác chưa
                    already_has_roi = False
                    for other_roi in rois:
                        if other_roi.get("assigned_employee") == emp.track_id and other_roi["id"] != roi_id:
                            already_has_roi = True
                            break
                    if already_has_roi:
                        continue

                    ex1, ey1, ex2, ey2 = emp.bbox
                    emp_box_poly = Polygon([(ex1, ey1), (ex2, ey1), (ex2, ey2), (ex1, ey2)])

                    if poly_obj.intersects(emp_box_poly):
                        inter_area = poly_obj.intersection(emp_box_poly).area
                        overlap_ratio = inter_area / roi_area if roi_area > 0 else 0

                        if overlap_ratio >= OVERLAP_THRESHOLD:
                            if overlap_ratio > max_overlap_ratio:
                                max_overlap_ratio = overlap_ratio
                                best_new_emp = emp

                # Xử lý bộ đếm frame để tránh hiện tượng nhấp nháy gán nhầm
                if best_new_emp is not None:
                    if roi_id not in best_new_emp.roi_frame_counters:
                        best_new_emp.roi_frame_counters[roi_id] = 0
                    
                    best_new_emp.roi_frame_counters[roi_id] += 1
                    if best_new_emp.roi_frame_counters[roi_id] >= M_FRAMES:
                        matched_emp = best_new_emp
                else:
                    for emp in employees:
                        if roi_id in emp.roi_frame_counters:
                            emp.roi_frame_counters[roi_id] = max(0, emp.roi_frame_counters[roi_id] - 1)

            # BƯỚC C: Cập nhật trạng thái và gán ID nhân viên vào ROI
            if matched_emp is not None:
                roi["assigned_employee"] = matched_emp.track_id
                roi["state"] = "occupied"
                assigned_emp_ids_this_frame.add(matched_emp.track_id)
            else:
                if current_assigned_emp is not None:
                    roi["state"] = "missing"
                else:
                    roi["assigned_employee"] = None
                    roi["state"] = "absence"

        # 4. VẼ GIAO DIỆN (VISUALIZATION)
        skeleton_pairs = base_config["pose_skeleton"]
        
        # Vẽ nhân viên và ID tracking
        for emp in employees:
            x1, y1, x2, y2 = [int(v) for v in emp.bbox]
            cv2.rectangle(vid_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            assigned_roi_str = "None"
            for roi in rois:
                if roi.get("assigned_employee") == emp.track_id:
                    assigned_roi_str = roi["id"]
                    break

            label = f"ID: {emp.track_id} | ROI: {assigned_roi_str}"
            cv2.putText(vid_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            valid_kpts = {}
            for idx, kp in enumerate(emp.keypoints):
                kx, ky, kconf = kp
                if kconf > 0.3:
                    valid_kpts[idx] = (int(kx), int(ky))
                    cv2.circle(vid_frame, (int(kx), int(ky)), 3, (0, 0, 255), -1)

            for pointA, pointB in skeleton_pairs:
                if pointA in valid_kpts and pointB in valid_kpts:
                    cv2.line(vid_frame, valid_kpts[pointA], valid_kpts[pointB], (255, 0, 0), 2)

        # Vẽ các khu vực ROI với màu sắc tương ứng theo state
        for roi in rois:
            polygon = np.array(roi["polygon"], np.int32)
            state = roi["state"]
            
            if state == "missing":
                color = tuple(base_config["red"])     # Nhân viên bỏ vị trí
            elif state == "occupied":
                color = tuple(base_config["green"])   # Đang có nhân viên ngồi làm việc
            elif state == "absence":
                color = tuple(base_config["gray"])    # Vùng trống / vắng mặt từ đầu
            else:
                color = tuple(base_config["blue"])

            cv2.polylines(vid_frame, [polygon], isClosed=True, color=color, thickness=2)
            
            display_text = f"{roi['id']}"
            if roi.get("assigned_employee"):
                display_text += f" (Emp: {roi['assigned_employee']})"
            
            cv2.putText(vid_frame, display_text, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Staff Absence & Behavior Tracking", vid_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
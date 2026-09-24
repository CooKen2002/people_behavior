import cv2
import onnxruntime as ort

from ...utils.yaml_utils import load_yaml
from ...modules.hand_washing.lavabo_roi import get_handroi, select_roi_points_from_image
from ...core.frame import Frame

base_config = load_yaml("configs/base.yaml")
config = load_yaml("configs/hand_washing.yaml")

image_path = config["image_path"]
video_path = config["video_path"]

# Khởi tạo session cho Object Detection (bồn rửa) và Pose Estimation (khớp xương người)
obj_session = ort.InferenceSession(config["obj_model_path"], providers=base_config["providers"])
pose_session = ort.InferenceSession(config["pose_model_path"], providers=base_config["providers"])

capCam = cv2.VideoCapture(video_path)

if not capCam.isOpened():
    print("Error: Can't open cam or video.")
    exit()

if __name__ == "__main__":
    # 1. Nhận diện ROI bồn rửa tự động hoặc thủ công
    origin_image = cv2.imread(image_path)
    origin_frame = Frame(origin_image, base_config['input_size'])
    
    rois = get_handroi(origin_frame)
    if not rois:
        print("Không tìm thấy bồn rửa tự động, vui lòng chọn thủ công...")
        rois = select_roi_points_from_image(origin_frame)

    while capCam.isOpened():
        ret, frame = capCam.read()
        if not ret:
            print("CLOSE")
            break

        # Đóng gói frame hiện tại vào lớp Frame để xử lý chung
        current_frame = Frame(frame, base_config['input_size'])

        # 2. Vẽ các vùng ROI (bồn rửa) lên khung hình
        for roi in rois:
            left, top, width, height = roi.bbox  # Đúng thứ tự [left, top, width, height]
            cv2.rectangle(
                frame,
                (left, top),
                (left + width, top + height),
                base_config["green"],
                base_config["bbox_thickness"],
            )

        # 3. Chạy Pose Estimation để tìm khớp xương người
        pose_output = current_frame.infer(pose_session, mode="resize")
        indices, boxes, confidences, keypoints_list = current_frame.parse_pose_estimation(
            pose_output, 
            conf_threshold=config.get("pose_conf_threshold", 0.5), 
            nms_threshold=config.get("pose_nms_threshold", 0.4), 
            mode="resize"
        )

        # 4. Kiểm tra logic rửa tay: Cả 2 cổ tay đồng thời nằm trong bất kỳ ROI bồn rửa nào
        is_washing_active = False

        for kpts in keypoints_list:
            # Theo chuẩn COCO Keypoints: 9 là cổ tay trái (left wrist), 10 là cổ tay phải (right wrist)
            if len(kpts) > 10:
                left_wrist = kpts[9]     # [x, y, confidence]
                right_wrist = kpts[10]   # [x, y, confidence]

                for roi in rois:
                    if roi.is_washing(right_wrist, left_wrist):
                        is_washing_active = True
                        break
            if is_washing_active:
                break

        # 5. Hiển thị thông báo trạng thái lên màn hình
        if is_washing_active:
            cv2.putText(
                frame, 
                "Washing", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, 
                (0, 0, 255), 
                3
            )

        cv2.imshow("Hand Washing Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Giải phóng tài nguyên
    capCam.release()
    cv2.destroyAllWindows()
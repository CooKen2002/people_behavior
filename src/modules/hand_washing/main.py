import cv2

from ...utils.yaml_utils import load_yaml
from ...modules.hand_washing.lavabo_roi import get_handroi, select_roi_points_from_image
from ...core.frame import Frame

base_config = load_yaml("configs/base.yaml")
config = load_yaml("configs/hand_washing.yaml")

obj_model_path = config["obj_model_path"]
image_path = config["image_path"]
video_path = config["video_path"]

capCam = cv2.VideoCapture(video_path)

if not capCam.isOpened():
    print("Error: Can't open cam or video.")
    exit()

fps = capCam.get(cv2.CAP_PROP_FPS)
frame_width = int(capCam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capCam.get(cv2.CAP_PROP_FRAME_HEIGHT))

if __name__ == "__main__":

    if get_handroi(image_path) is None:
        rois = get_handroi(image_path)
    else:
        rois = select_roi_points_from_image(image_path)

    while capCam.isOpened():
        ret, frame = capCam.read()
        if not ret:
            print("CLOSE")
            break
        for roi in rois:
            top, left, width, height = roi.bbox
            confidence = roi.confidence
            cv2.rectangle(
                frame,
                (top, left),
                (top + width, left + height),
                base_config["green"],
                base_config["bbox_thickness"],
            )
        # Hiển thị frame đã vẽ (dùng original.frame thay vì frame)

        pose_frame = Frame(frame,base_config['input_size'])
        
        cv2.imshow("Example", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Giải phóng tài nguyên
    capCam.release()
    cv2.destroyAllWindows()

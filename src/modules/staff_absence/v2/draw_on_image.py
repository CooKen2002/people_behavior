import cv2
import numpy as np
            
from src.utils.json_utils import save_json

def select_multiple_rois(frame, window_name="Select Multiple ROIs"):
    """
    Cho phép chọn nhiều vùng ROI, mỗi vùng gồm 4 điểm.
    - Click chuột trái: Chọn điểm (tối đa 4 điểm cho mỗi vùng).
    - Phím 'c': Lưu vùng hiện tại và tiếp tục chọn vùng mới (sau khi đã chọn đủ 4 điểm).
    - Phím 'q': Thoát và lưu tất cả các vùng đã chọn.
    """
    all_rois = []          # Danh sách chứa tất cả các vùng ROI (mỗi vùng là 4 điểm)
    current_points = []    # 4 điểm của vùng đang chọn

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
            current_points.append((x, y))
            print(f"Đã chọn điểm {len(current_points)}/4 của vùng hiện tại: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        # Tạo bản sao của ảnh gốc để vẽ các vùng đã hoàn thành và vùng đang thao tác
        display = frame.copy()

        # 1. Vẽ lại tất cả các vùng ROI đã được lưu trước đó (màu xanh lá cây)
        for idx, roi in enumerate(all_rois):
            pts_np = np.array(roi, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
            # Đánh nhãn số thứ tự của vùng ROI
            M = cv2.moments(pts_np)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(display, f"ROI {idx+1}", (cX - 20, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 2. Vẽ các điểm và đường của vùng ROI đang chọn dở dang (màu đỏ/tím)
        for i, p in enumerate(current_points):
            cv2.circle(display, p, 6, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (p[0] + 10, p[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if len(current_points) > 1:
            pts_np = np.array(current_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts_np], isClosed=False, color=(255, 0, 255), thickness=2)

        if len(current_points) == 4:
            pts_np = np.array(current_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts_np], isClosed=True, color=(255, 0, 255), thickness=2)
            
            # Hướng dẫn khi đã đủ 4 điểm cho 1 vùng
            cv2.putText(
                display,
                "Da du 4 diem! Nhan 'c' de chon tiep, 'q' de hoan tat",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            # Hướng dẫn chung
            cv2.putText(
                display,
                f"Vung hien tai: {len(current_points)}/4 diem | Da chon {len(all_rois)} vung | 'q': Thoat",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        # Nếu bấm 'q': kết thúc và thoát
        if key == ord('q'):
            print("Đã hoàn tất quá trình chọn ROI.")
            break

        # Nếu đã đủ 4 điểm và người dùng bấm 'c': lưu vùng hiện tại và reset để chọn vùng tiếp theo
        if len(current_points) == 4 and key == ord('c'):
            all_rois.append(list(current_points))
            print(f"Đã lưu ROI thứ {len(all_rois)}")
            current_points = []  # Xóa để bắt đầu chọn vùng mới

    cv2.destroyWindow(window_name)
    return [np.array(roi, dtype=np.float32) for roi in all_rois]

# --- Đoạn code gọi hàm ---
image_path = "data/raw/1809_frame.jpg"
frame = cv2.imread(image_path)

if frame is not None:
    list_rois = select_multiple_rois(frame, window_name="Select Multiple ROIs")
    save_json("data/annotations/selected_rois.json", list_rois)
    print(f"\nTổng số vùng ROI đã chọn: {len(list_rois)}")
    for i, roi in enumerate(list_rois):
        print(f"ROI {i+1}:\n{roi}\n")
else:
    print(f"Không đọc được ảnh tại đường dẫn: {image_path}")
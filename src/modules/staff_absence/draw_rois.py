import cv2
import numpy as np

from ...core.roi import PolygonRoi

def select_multiple_rois(frame, window_name="Select Multiple ROIs"):
    """
    Cho phép chọn nhiều vùng ROI, mỗi vùng gồm 4 điểm.
    - Click chuột trái: Chọn điểm (tối đa 4 điểm cho mỗi vùng).
    - Phím 'c': Lưu vùng hiện tại (sẽ hỏi tên ROI qua terminal) và tiếp tục chọn vùng mới.
    - Phím 'q': Thoát và lưu tất cả các vùng đã chọn.

    Trả về: list[dict] — mỗi dict có dạng {"roi_id": str, "polygon": [[x,y], ...]}
    """
    all_rois = []          # list các dict {"roi_id":..., "polygon":...}
    current_points = []    # 4 điểm của vùng đang chọn

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
            current_points.append((x, y))
            print(f"Đã chọn điểm {len(current_points)}/4 của vùng hiện tại: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = frame.copy()

        # 1. Vẽ lại tất cả các vùng ROI đã lưu, hiển thị đúng roi_id thay vì số thứ tự
        for roi in all_rois:
            pts_list = roi.polygon
            pts_np = np.array(pts_list, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
            M = cv2.moments(pts_np)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(display, roi.id, (cX - 20, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 2. Vẽ điểm/đường của vùng đang chọn dở dang
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
            cv2.putText(
                display,
                "Da du 4 diem! Nhan 'c' de chon tiep, 'q' de hoan tat",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        else:
            cv2.putText(
                display,
                f"Vung hien tai: {len(current_points)}/4 diem | Da chon {len(all_rois)} vung | 'q': Thoat",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Đã hoàn tất quá trình chọn ROI.")
            break

        if len(current_points) == 4 and key == ord('c'):
            # input() sẽ tạm dừng vòng lặp cv2 chờ gõ terminal — đúng ý đồ,
            # vì lúc này cửa sổ đã hiển thị đủ 4 điểm, không cần thao tác chuột thêm.
            default_id = "None" # f"ROI_{len(all_rois) + 1}"
            typed = input(f"Tên ROI (Enter để dùng mặc định '{default_id}'): ").strip()
            current_roi = PolygonRoi(id=typed if typed else default_id, state= "", polygon=current_points)
            all_rois.append(current_roi)

            # all_rois.append({
            #     "roi_id": roi_id,
            #     "polygon": [list(p) for p in current_points],
            # })
            print(f"Đã lưu ROI {current_roi.id}")
            current_points = []

    cv2.destroyWindow(window_name)
    return all_rois


if __name__ == "__main__":
    image_path = "data/raw/1809_frame.jpg"
    frame = cv2.imread(image_path)

    if frame is not None:
        list_rois = select_multiple_rois(frame, window_name="Select Multiple ROIs")
        print(f"\nTổng số vùng ROI đã chọn: {len(list_rois)}")
        for roi in list_rois:
            print(f"{roi.id}:\n{roi.polygon}\n")
    else:
        print(f"Không đọc được ảnh tại đường dẫn: {image_path}")
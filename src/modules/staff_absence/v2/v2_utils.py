import cv2
import numpy as np


def HomoGraphyTransform(src_pts, cam_pts, map_pts):
    """Tính toán ma trận Homography từ các điểm trên sơ đồ (src_pts) sang camera (cam_pts)

    và chuyển đổi danh sách các tọa độ trên sơ đồ thành tọa độ tương ứng trên
    camera.
    """
    # 1. Chuyển đổi các điểm sang kiểu numpy array float32 chuẩn xác
    src_pts = np.array(src_pts, dtype=np.float32)
    cam_pts = np.array(cam_pts, dtype=np.float32)

    # 2. Tính ma trận Homography H
    H, _ = cv2.findHomography(src_pts, cam_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError(
            "Không thể tính toán ma trận Homography. Kiểm tra lại các điểm truyền"
            " vào!"
        )

    map_cam_pts = []

    for item in map_pts:
        map_cam_pt = item.copy()
        original_coords = map_cam_pt[
            "cordinates"
        ]  # Danh sách 4 điểm: [[x1, y1], [x2, y2], ...]

        new_table_coords = []
        for pt in original_coords:
            # Định dạng lại từng điểm thành mảng 3 chiều (1, 1, 2)
            pt_reshaped = np.array([[[pt[0], pt[1]]]], dtype=np.float32)

            # Biến đổi tọa độ qua ma trận H
            transformed_pt = cv2.perspectiveTransform(pt_reshaped, H)

            # Lấy kết quả [x, y]
            new_coords = transformed_pt[0][0]
            new_table_coords.append([float(new_coords[0]), float(new_coords[1])])

        # Cập nhật lại 4 điểm góc mới sau khi ánh xạ
        map_cam_pt["cordinates"] = new_table_coords
        map_cam_pts.append(map_cam_pt)

    return map_cam_pts


def draw_camera_maps(image_path, map_cam_pts):
    # 1. Đọc ảnh camera gốc
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc được ảnh từ đường dẫn: {image_path}")
        return

    output_image = image.copy()

    # 3. Duyệt qua từng bàn trong danh sách để vẽ
    for item in map_cam_pts:
        table_id = item.get("table_id", "")
        employee = item.get("employee", "")
        coords = item.get("cordinates", [])  # Danh sách 4 điểm [x, y] trên camera

        if not coords or len(coords) < 4:
            continue

        # Chuyển đổi định dạng tọa độ sang mảng numpy kiểu int (OpenCV yêu cầu để vẽ)
        pts = np.array(coords, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Chọn màu sắc: Nếu bàn có nhân viên ngồi -> màu xanh lá, bàn trống -> màu đỏ
        color = (0, 255, 0) if employee else (0, 0, 255)

        # Vẽ đường viền 4 cạnh của bàn
        cv2.polylines(output_image, [pts], isClosed=True, color=color, thickness=2)

        # (Tùy chọn) Tô màu mờ bên trong vùng bàn để dễ quan sát
        overlay = output_image.copy()
        cv2.fillPoly(overlay, [pts], color)
        output_image = cv2.addWeighted(overlay, 0.2, output_image, 0.8, 0)

        # Hiển thị Table ID lên góc trên của bàn
        first_point = (int(coords[0][0]), int(coords[0][1]))
        label = f"Table {table_id}"
        if employee:
            label += f" ({employee})"

        cv2.putText(
            output_image,
            label,
            (first_point[0], first_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Chấm các điểm góc nhỏ trên bàn
        for pt in coords:
            cv2.circle(
                output_image, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1
            )  # Chấm xanh dương ở các góc

    # 4. Hiển thị ảnh kết quả
    cv2.imshow("Camera Table Mapping", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

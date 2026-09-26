## 1. TỔNG QUAN HỆ THỐNG & MÔ HÌNH PHÒNG THI

### Quy chuẩn phòng thi: 24 thí sinh/phòng
* **Ưu tiên 1:** 24 bàn ghế đơn tách biệt.
* **Ưu tiên 2:** 12 bàn dài (mỗi bàn 2 học sinh ngồi ở 2 đầu/góc).

### Kiến trúc phân tán (Edge-Cloud Architecture)
* **Edge Device (Thiết bị tại phòng thi):** Chạy luồng AI xử lý hình ảnh từ Multi-camera, thực thi thuật toán phát hiện, phân tích hành vi và đóng gói dữ liệu.
* **Web Server (Cloud/Local Server):** Nhận dữ liệu trạng thái qua WebSocket/MQTT, render giao diện giám sát Realtime (Sơ đồ lưới 24 vị trí + Video Stream).

---

## 2. PHÂN TÍCH CÁC MODULE CHỨC NĂNG

### Module 1: Detect Examinee Region (DER - Định vị & Trạng thái vị trí ngồi)
* **Mô tả:** Tự động hoặc cấu hình cố định phạm vi bàn học, đánh số thứ tự vị trí (`desk_id`) và kiểm tra trạng thái từng ô (`has_student: true/false`).
* **Giải pháp kỹ thuật tối ưu (Giải quyết vấn đề độ tin cậy của model detect bàn):**
  * Sử dụng **Static ROI Mapping (Vùng quan tâm tĩnh)** thay vì bắt mô hình AI predict bàn liên tục. Vị trí bàn học trong phòng thi là cố định, do đó tọa độ các ROI (`desk_id` từ 1 đến 24) sẽ được cấu hình một lần từ đầu (qua Web UI hoặc file config).
  * Mô hình AI chỉ tập trung detect **Examinee (Học sinh)** và **Objects (Điện thoại/Thiết bị cấm)**.
  * Thuật toán không gian (Spatial Mapping) kiểm tra tâm (center point) hoặc diện tích giao nhau (IoU / Intersection) của BBox học sinh với từng ROI tĩnh để gán trạng thái (`state`).

### Module 2: Advanced Cheat Detection (Phân tích hành vi gian lận)
* **Analys Examinee Behavior (Phân tích hành vi thí sinh):**
  * *Leaving (Rời vị trí):* Phát hiện khi vị trí ROI trống (`has_student = false`) trong khoảng thời gian vượt ngưỡng cho phép.
  * *Leaning (Quay cóp/Nghiêng người):* Theo dõi độ lệch tâm của học sinh so với ROI bàn định sẵn hoặc sử dụng Light Pose Estimation cho các vùng nghi ngờ để phát hiện hành vi quay đầu, cúi quá thấp.
  * *Using another devices (Sử dụng thiết bị cấm):* Object Detection nhận diện điện thoại, thiết bị thu phát sóng trong khu vực bàn học.
  * *Suspicious Behavior (Hành vi bất thường khác):* Các cử động lặp lại hoặc bất thường.
* **Detect Suspicious Intervention (Phát hiện tác động bên ngoài):**
  * *Analys Examiner Behavior:* Cảnh báo khi giám thị tiếp cận/tập trung lâu tại một khu vực bất thường.
  * *Detect Extern intervention:* Phát hiện người lạ xâm nhập phòng thi hoặc có hành vi trao đổi đồ vật từ bên ngoài vào.

### Module 3: Multi-Camera Synchronization (Đồng bộ dữ liệu đa góc nhìn)
* Giải quyết bài toán khuất tầm nhìn (occlusion) trong phòng 24 thí sinh bằng hệ thống từ 2 camera trở lên.
* **Dữ liệu đồng bộ:** Đồng bộ toàn bộ trạng thái không gian bàn học (`room_id`, `desk_id`, `state`, `timestamp`) giữa các luồng camera về thiết bị xử lý trung tâm.

### Module 4: Realtime Web Server & Dashboard
* **Edge to Server Communication:** Đóng gói dữ liệu dạng JSON (`room_id`, `desk_id`, `state`, `warning_type`, `timestamp`) truyền qua **WebSocket** hoặc **MQTT** đảm bảo độ trễ thấp ($< 500\text{ms}$).
* **Frontend Dashboard:**
  * **Sơ đồ phòng thi (Grid 24 ô):** Thay đổi màu sắc trực quan theo trạng thái (Xanh: Bình thường, Đỏ: Cảnh báo vắng/gian lận, Vàng: Cảnh báo hành vi).
  * **Live Stream:** Hiển thị luồng video trực tiếp từ camera phòng thi qua **WebRTC** hoặc **RTSP-to-Web** (tích hợp MediaMTX / go2rtc).

---

## 3. ĐỀ XUẤT PIPELINE TRIỂN KHAI THỰC TẾ (HYBRID PIPELINE)

Để giải quyết bài toán tải trọng tính toán (**Compute Load**) trên thiết bị biên, hệ thống áp dụng luồng xử lý tối ưu sau:

1. **Khởi tạo (Initialization):** 
   * Nạp file cấu hình chứa tọa độ tĩnh của 24 ROI bàn học (`desk_id`) tương ứng với khung hình camera.
2. **Inference (Chạy vòng lặp thời gian thực):** 
   * Sử dụng mô hình Object Detection gọn nhẹ (ví dụ: YOLOv11) để nhận diện `Student` và `Forbidden_Device`.
3. **Post-processing & Mapping:**
   * Áp dụng thuật toán **ByteTrack / BoT-SORT** để giữ ổn định ID học sinh qua các khung hình.
   * Chiếu tọa độ BBox học sinh vào danh sách Static ROIs để cập nhật `state` (`has_student`) cho từng `desk_id`.
   * Kiểm tra các điều kiện vi phạm hành vi (Leaning, Leaving, Using Device) dựa trên logic tọa độ và thời gian (`time-threshold`).
4. **Data Sync & Publish:** 
   * Đóng gói payload JSON đẩy qua WebSocket/MQTT lên Web Server để cập nhật UI Dashboard và phát cảnh báo âm thanh/visual nếu có sự cố.
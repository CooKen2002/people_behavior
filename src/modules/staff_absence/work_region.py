"""
work_region.py
Kết hợp ROI (hình học thuần, từ roi.py) với domain logic gán nhân viên
vào khu vực làm việc.

Khác với logic dict-based trong main.py gốc, class này thêm 2 điểm:

1. Theo dõi ĐÚNG person_id (track_id) đang chiếm khu vực, thay vì chỉ
   kiểm tra "có AI ĐÓ trong ROI hay không". Nếu không lưu định danh, 2
   người đi ngang qua nhau trong cùng 1 ROI có thể khiến region bị gán
   nhầm sang track khác mà không ai biết — nhất là với layout bàn sát
   nhau.

2. Vá bug: trong main.py gốc, một khi region đã chuyển sang "away",
   KHÔNG có đường nào đưa nó quay lại "unassigned" nếu người đó biến
   mất hẳn (đi về, nghỉ việc, tắt máy...). Region sẽ kẹt ở "away" vĩnh
   viễn. Class này thêm missing_threshold: nếu person_id không còn
   xuất hiện ở BẤT KỲ ĐÂU trong khung hình (không riêng trong ROI) đủ
   lâu, mới reset về unassigned.

Chưa đưa vào: phân loại tư thế ngồi/đứng (posture classification qua
tỉ lệ hip-knee/torso đã bàn trước đó) — main.py hiện tại chỉ xét
"hip có nằm trong ROI hay không", không phân biệt đang ngồi hay đứng
tại chỗ. Đây là hướng mở rộng tự nhiên tiếp theo nếu cần, không bắt
buộc phải có ngay để refactor lần này chạy đúng.
"""

from typing import Optional

from ...core.roi import ROI


class WorkRegion:
    STATE_ABSENCE = "absence"
    STATE_UNASSIGNED = "unassigned"
    STATE_ASSIGNED = "assigned"
    STATE_AWAY = "away"

    def __init__(
        self,
        region_id: str,
        roi: ROI,
        employee_name: str = "",
        employee_state: str = "present",
        assign_threshold: int = 5,
        away_threshold: int = 15,
        missing_threshold: int = 90,
    ):
        self.region_id = region_id
        self.roi = roi
        self.employee_name = employee_name or region_id
        self.employee_state = employee_state

        # missing_threshold nên đặt >= TRACK_MAX_DISAPPEARED của SimpleTracker.
        # Lý do: SimpleTracker tự giữ track_id sống qua occlusion ngắn (dưới
        # max_disappeared), chỉ khi tracker THỰC SỰ deregister track_id đó,
        # người này mới hoàn toàn biến mất khỏi detected_people mỗi frame.
        # Đặt missing_threshold nhỏ hơn max_disappeared là vô nghĩa vì điều
        # kiện "not still_in_frame" chỉ có thể đúng sau khi tracker đã
        # deregister rồi.
        self.assign_threshold = assign_threshold
        self.away_threshold = away_threshold
        self.missing_threshold = missing_threshold

        self.state = self.STATE_ABSENCE if employee_state == "absence" else self.STATE_UNASSIGNED
        self.person_id: Optional[str] = None

        self._present_counter = 0
        self._absence_counter = 0
        self._missing_counter = 0

    @classmethod
    def from_dict(
        cls,
        data: dict,
        assign_threshold: int,
        away_threshold: int,
        missing_threshold: int,
    ) -> "WorkRegion":
        roi = ROI(tuple(data["bbox_xywh"]))
        return cls(
            region_id=data["region_id"],
            roi=roi,
            employee_name=data.get("employee_name", data["region_id"]),
            employee_state=data.get("employee_state", "present"),
            assign_threshold=assign_threshold,
            away_threshold=away_threshold,
            missing_threshold=missing_threshold,
        )

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "bbox_xywh": list(self.roi.bbox_xywh()),
            "employee_name": self.employee_name,
            "employee_state": self.employee_state,
            "region_state": self.state,
            "track_id": "" if self.person_id is None else str(self.person_id),
        }

    def set_employee_state(self, employee_state: str) -> None:
        """Gọi khi hệ thống chấm công/lịch làm việc báo nhân viên nghỉ hay
        đi làm — tách biệt hoàn toàn khỏi state do camera tự suy ra."""
        self.employee_state = employee_state
        if employee_state == "absence":
            self.state = self.STATE_ABSENCE
            self.person_id = None
            self._present_counter = 0
            self._absence_counter = 0
            self._missing_counter = 0
        elif self.state == self.STATE_ABSENCE:
            self.state = self.STATE_UNASSIGNED

    def contains_point(self, x: float, y: float) -> bool:
        rx, ry, rw, rh = self.roi.bbox_xywh()
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def update(self, detected_people: list) -> None:
        """
        detected_people: list các dict {"track_id", "box", "hip", "keypoints"}
        đúng format main.py đang build mỗi frame. Gọi đúng 1 lần/frame/region.
        """
        if self.employee_state == "absence":
            return

        people_in_region = [
            p for p in detected_people
            if p["hip"][0] is not None and self.contains_point(*p["hip"])
        ]

        matched = None
        if self.person_id is not None:
            matched = next((p for p in people_in_region if p["track_id"] == self.person_id), None)
        if matched is None and people_in_region:
            matched = people_in_region[0]

        has_someone = matched is not None
        still_in_frame = self.person_id is not None and any(
            p["track_id"] == self.person_id for p in detected_people
        )

        if self.state == self.STATE_UNASSIGNED:
            if has_someone:
                self._present_counter += 1
                if self._present_counter >= self.assign_threshold:
                    self.state = self.STATE_ASSIGNED
                    self.person_id = matched["track_id"]
                    self._absence_counter = 0
                    self._missing_counter = 0
            else:
                self._present_counter = 0
            return

        # Từ đây trở xuống (ASSIGNED / AWAY): region đã có chủ (self.person_id)
        if has_someone and matched["track_id"] == self.person_id:
            self._absence_counter = 0
            self._missing_counter = 0
            if self.state == self.STATE_AWAY:
                self.state = self.STATE_ASSIGNED
            return

        # Không thấy ĐÚNG người trong ROI ở frame này
        if self.state == self.STATE_ASSIGNED:
            self._absence_counter += 1
            if self._absence_counter >= self.away_threshold:
                self.state = self.STATE_AWAY

        # ---- Fix bug: đường timeout còn thiếu trong bản gốc ----
        if not still_in_frame:
            self._missing_counter += 1
            if self._missing_counter >= self.missing_threshold:
                self._reset_to_unassigned()
        else:
            self._missing_counter = 0

    def _reset_to_unassigned(self) -> None:
        self.state = self.STATE_UNASSIGNED
        self.person_id = None
        self._present_counter = 0
        self._absence_counter = 0
        self._missing_counter = 0

    def display_color(self):
        return {
            self.STATE_ASSIGNED: (0, 255, 0),
            self.STATE_AWAY: (0, 255, 255),
            self.STATE_UNASSIGNED: (255, 0, 0),
            self.STATE_ABSENCE: (128, 128, 128),
        }.get(self.state, (128, 128, 128))

    def label(self) -> str:
        return f"{self.region_id} | {self.employee_name} | {self.state}"

import numpy as np
from typing import List
from src.core.human import Human


class GeneralTracker:
    def __init__(self, max_missed_frames: int = 30, iou_threshold: float = 0.3):
        """
        :param max_missed_frames: Số frame tối đa giữ lại track khi người đó biến mất (tránh mất ID khi model detect sót vài frame).
        :param iou_threshold: Ngưỡng IoU tối thiểu để coi là cùng một đối tượng giữa 2 frame liên tiếp.
        """
        self.next_id = 1
        self.tracks: List[Human] = []
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold

    def _compute_iou(self, boxA, boxB):
        """Hàm phụ trợ tính IoU giữa 2 bounding box [x1, y1, x2, y2]"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, detections: List[List[float]]) -> List[Human]:
        """
        Cập nhật tracker với danh sách các detections mới từ model ở frame hiện tại.
        :param detections: List các bbox dạng [[x1, y1, x2, y2], ...]
        :return: Danh sách các đối tượng Human đang active (đã được gán ID ổn định)
        """
        # Nếu chưa có track nào, khởi tạo toàn bộ detections hiện tại thành các track mới
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Human(track_id=self.next_id, bbox=det))
                self.next_id += 1
            return self.tracks

        # Nếu frame hiện tại không phát hiện ra ai, tăng biến đếm missed cho tất cả các track cũ
        if len(detections) == 0:
            for track in self.tracks:
                track.mark_missed()
            # Lọc bỏ các track đã mất quá số frame cho phép
            self.tracks = [
                t for t in self.tracks if t.time_since_update <= self.max_missed_frames
            ]
            return self.tracks

        # Tính ma trận IoU giữa các track hiện tại và các detections mới
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._compute_iou(track.bbox, det)

        # Thuật toán matching đơn giản dựa trên Greedy IoU
        matched_tracks = set()
        matched_dets = set()

        # Sắp xếp các cặp có IoU từ cao xuống thấp
        while True:
            if iou_matrix.size == 0:
                break
            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[t_idx, d_idx]

            if max_iou < self.iou_threshold:
                break  # Nếu IoU cao nhất vẫn nhỏ hơn ngưỡng thì dừng

            if t_idx not in matched_tracks and d_idx not in matched_dets:
                # Khớp track cũ với detection mới
                self.tracks[t_idx].update(detections[d_idx])
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

            # Xóa cặp này khỏi ma trận để xét tiếp
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        # Xử lý các track cũ không tìm thấy match ở frame này
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()

        # Xử lý các detection mới chưa được match (tạo track mới hoàn toàn)
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append(Human(track_id=self.next_id, bbox=det))
                self.next_id += 1

        # Dọn dẹp: Xóa bỏ các track đã biến mất quá lâu
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_missed_frames
        ]

        return self.tracks

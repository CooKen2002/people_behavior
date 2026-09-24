import cv2
import numpy as np
from typing import Generator, Tuple, Optional, Union


class VideoLoader:
    def __init__(
        self, source: Union[str, int] = 0, target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Khởi tạo video loader.
        :param source: Đường dẫn file video, link RTSP string, hoặc ID webcam (ví dụ: 0).
        :param target_size: Kích thước mong muốn (width, height) để resize từng frame.
        """
        self.source = source
        self.target_size = target_size
        self.cap = None

    def __enter__(self):
        """Hỗ trợ sử dụng với cú pháp `with VideoLoader(...) as loader:`"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ConnectionError(f"Không thể mở nguồn video/camera: {self.source}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Tự động giải phóng tài nguyên khi thoát khối with."""
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def get_metadata(self) -> dict:
        """Lấy thông số cơ bản của video gốc."""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Video chưa được mở.")
        return {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """Generator trả về từng frame liên tục phục vụ cho vòng lặp AI pipeline."""
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Video chưa được mở.")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break  # Hết video hoặc mất kết nối camera

            if self.target_size:
                frame = cv2.resize(frame, self.target_size)

            yield frame

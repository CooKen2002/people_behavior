import os
import cv2
import numpy as np
from typing import Generator, Tuple, Optional, Union

class ImageLoader:
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Khởi tạo image loader.
        :param target_size: Kích thước mong muốn (width, height) để resize ảnh.
        """
        self.target_size = target_size

    def load_single(self, image_path: str) -> np.ndarray:
        """Đọc một ảnh đơn từ đường dẫn."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể giải mã file ảnh (có thể file bị lỗi): {image_path}")
            
        if self.target_size:
            img = cv2.resize(img, self.target_size)
        return img

    def load_from_folder(self, folder_path: str) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Generator duyệt qua tất cả các ảnh trong thư mục (hỗ trợ .jpg, .png, .jpeg)."""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        if not os.path.exists(folder_path):
            raise NotADirectoryError(f"Thư mục không tồn tại: {folder_path}")

        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = self.load_single(img_path)
                    yield filename, img
                except Exception as e:
                    print(f"Lỗi khi đọc file {filename}: {e}")
import json
import os   
import numpy as np

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(json_path, data):
    # Tự động tạo thư mục chứa file nếu chưa tồn tại (ví dụ: data/annotated/)
    dir_name = os.path.dirname(json_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    # Chuyển đổi numpy array thành list chuẩn Python để serialize sang JSON được
    # (vì numpy float32 không tự động serialize được qua json.dump)
    if isinstance(data, list):
        data = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in data]

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
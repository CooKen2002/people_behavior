# Human Behavior Modules

Install venv + dependencies (current version: Python 3.10.11)
```bash
    py -m venv .venv
    pip install -r requirements.txt
```

Repo Structure
```bash
    human-behavior/
    │
    ├── .github/                  # CI/CD workflows, issue templates
    │   └── workflows/
    │       ├── ci.yml            # Test tự động (pytest)
    │       └── lint.yml          # Kiểm tra định dạng code (flake8, black)
    │
    ├── configs/                  # File cấu hình dạng YAML cho từng domain/model
    │
    ├── data/                     # Thư mục lưu dữ liệu (thường ignore qua .gitignore)
    │   ├── raw/                  # Video hoặc frame thô ban đầu
    │   ├── processed/            # Dữ liệu đã chuẩn hóa / cắt đoạn
    │   └── annotations/          # File nhãn (JSON, CSV, YOLO format txt)
    │
    ├── docs/                     # Tài liệu hướng dẫn, sơ đồ kiến trúc hệ thống
    │
    ├── models/                   # Trọng số mô hình đã train (weights)
    │
    ├── notebooks/                # Jupyter Notebooks phục vụ EDA và thử nghiệm nhanh
    │
    ├── src/                      # Source code chính của dự án
    │   ├── __init__.py
    │   │
    │   ├── core/                 # Các module nền tảng dùng chung (Shared Core)
    │   │
    │   ├── modules/              # Các Domain cụ thể (Mỗi bài toán là một gói module riêng)
    │   │   ├── __init__.py
    │   │   ├── exam_cheating/    # Modules 1: Phát hiện gian lận phòng thi
    │   │   ├── hand_washing/     # Modules 2: Kiểm tra quy trình rửa tay y tế/thực phẩm
    │   │   └── staff_absence/    # Modules 3: Giám sát nhân viên vắng mặt / rời vị trí
    │   │
    │   └── utils/                # Các tiện ích hỗ trợ chung
    │
    ├── tests/                    # Unit tests bằng pytest cho từng module
    ├── .gitignore                # Các file/thư mục không đưa lên Git (vram, weights, data)
    ├── Dockerfile                # Đóng gói môi trường chạy Docker
    ├── LICENSE                   # Giấy phép mã nguồn mở (MIT, Apache...)
    ├── README.md                 # Giới thiệu tổng quan dự án, hướng dẫn cài đặt
    └── requirements.txt          # Danh sách thư viện Python (ultralytics, opencv, mediapipe...)
```
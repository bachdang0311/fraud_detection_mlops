fraud_ml_ops_project/
├── data/
│   ├── raw/                     # Nơi chứa file CSV gốc (train_transaction.csv, ...)
│   └── processed/               # Nơi chứa dữ liệu đã qua tiền xử lý
├── notebooks/
│   └── fraud_ml_ops.ipynb       # Notebook gốc của bạn (để tham khảo)
├── src/
│   ├── __init__.py
│   ├── config.py                # Chứa các cấu hình (đường dẫn, tham số model)
│   ├── data_preprocessing.py    # Chứa các hàm tiền xử lý dữ liệu
│   ├── feature_engineering.py   # Chứa các hàm tạo đặc trưng
│   ├── train.py                 # Chứa hàm huấn luyện và đánh giá model
│   ├── predict.py               # (Tùy chọn) Chứa hàm dự đoán với model đã huấn luyện
│   └── utils.py                 # Chứa các hàm tiện ích (ví dụ: reduce_memory_usage)
├── models/                      # Nơi lưu trữ các model đã huấn luyện (ví dụ: lgbm_model.pkl)
├── artifacts/                   # Nơi lưu các thông tin khác (ví dụ: model_info.json, feature_importance.png)
├── tests/                       # Chứa các unit test (nếu có)
├── requirements.txt             # Danh sách các thư viện cần thiết
├── Dockerfile                   # (Tùy chọn) Để đóng gói ứng dụng
└── README.md
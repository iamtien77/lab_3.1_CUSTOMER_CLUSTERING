"""
=============================================================================
MODULE: NẠP VÀ PHÂN CHIA DỮ LIỆU (DATA LOADER)
=============================================================================
Mục đích:
- Đọc file dữ liệu thô (Mall_Customers.csv) từ thư mục data/raw/.
- Kiểm tra tính toàn vẹn cấu trúc (schema validation).
- Phân chia tập Train / Test và lưu trữ vào thư mục data/processed/.
=============================================================================
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from src.config import (
    DEFAULT_DATA_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT_RATIO,
    ID_COLUMN,
    AGE_COLUMN,
    INCOME_COLUMN,
    SPENDING_COLUMN,
)


def load_raw_data(file_path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Ý LÀM:
    1. Kiểm tra sự tồn tại của file dữ liệu (file_path hoặc DEFAULT_DATA_PATH).
    2. Đọc file bằng pandas (hỗ trợ .csv, .xlsx, .tsv).
    3. Chuẩn hóa tên các cột (xóa khoảng trắng thừa .str.strip()).
    4. Nếu có cột 'Genre' thì đổi tên thành 'Gender' cho chuẩn tiếng Anh.
    5. Trả về DataFrame dữ liệu thô.
    """
    pass


def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ý LÀM:
    1. Định nghĩa danh sách các cột bắt buộc: [CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)].
    2. Kiểm tra xem có cột nào bị thiếu so với kỳ vọng hay không.
    3. Kiểm tra số lượng dòng trùng lặp (df.duplicated().sum()).
    4. Kiểm tra mã khách hàng (CustomerID) có bị trùng lặp không (bắt buộc phải unique).
    5. Đếm tổng số giá trị bị khuyết (NaN/Null) trên toàn bảng.
    6. Trả về dictionary tóm tắt trạng thái hợp lệ (is_valid: True/False, chi tiết số dòng, số cột,...).
    """
    pass


def split_data(
    df: pd.DataFrame,
    test_size: float = TRAIN_TEST_SPLIT_RATIO,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ý LÀM:
    1. Phân chia tập dữ liệu thành Train (80%) và Holdout Test (20%) bằng train_test_split.
       - Mục đích: Tránh Data Leakage trước khi thực hiện Feature Scaling hay Huấn luyện mô hình.
    2. Reset index cho cả 2 tập để tránh xáo trộn chỉ số dòng.
    3. Lưu train_df vào data/processed/train.csv.
    4. Lưu test_df vào data/processed/test.csv.
    5. Lưu toàn bộ tập vào data/processed/full_customers.csv.
    6. Trả về tuple (train_df, test_df).
    """
    pass

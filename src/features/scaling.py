"""
=============================================================================
MODULE: CHUẨN HÓA ĐẶC TRƯNG (FEATURE SCALING)
=============================================================================
Mục đích (Đáp ứng Bước 2 trong đề bài Practice Exercise 3):
- Chuẩn hóa (Normalize) hoặc Tỉ lệ hóa chuẩn mực (Standardize) các đặc trưng:
  Age, Annual Income, Spending Score để đưa chúng về cùng một thang đo.
- Tránh tình trạng biến có giá trị lớn (như Thu nhập: 15-137) chi phối khoảng cách Euclidean
  so với các biến nhỏ hơn trong các thuật toán phân cụm.
- Hỗ trợ đóng gói Scikit-Learn Pipeline (fit trên Train, transform trên Test / khách hàng mới).
=============================================================================
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.config import NUMERICAL_FEATURES


class FeatureScaler:
    """
    Lớp chuẩn hóa đặc trưng đa năng cho phân cụm khách hàng.
    """

    def __init__(
        self,
        method: str = "standard",
        feature_cols: Optional[List[str]] = None,
        encode_gender: bool = False,
    ):
        """
        Ý LÀM:
        1. Nhận vào tham số 'method':
           - 'standard': StandardScaler (z = (x - mean) / std) -> Đưa về Mean=0, Std=1.
           - 'minmax': MinMaxScaler (x' = (x - min) / (max - min)) -> Đưa về khoảng [0, 1].
           - 'robust': RobustScaler (dựa vào Median và IQR) -> Kháng Outlier tốt nhất.
        2. Khởi tạo đối tượng scaler tương ứng từ scikit-learn.
        3. Lưu danh sách các cột cần scale (feature_cols, mặc định: Age, Annual Income, Spending Score).
        4. Cờ encode_gender: nếu True, mã hóa thêm cột Gender (Male=1, Female=0).
        """
        pass

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """
        Ý LÀM:
        1. Trích xuất các cột số feature_cols từ df (tập Train).
        2. Gọi self.scaler.fit() để học các tham số thống kê (Mean, Std hoặc Min, Max) CHỈ trên tập Train.
        3. Đánh dấu trạng thái đã fit.
        4. Trả về chính đối tượng (self).
        """
        pass

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Ý LÀM:
        1. Kiểm tra xem scaler đã được fit chưa (nếu chưa thì báo lỗi).
        2. Gọi self.scaler.transform() lên các cột số feature_cols của df.
        3. Nếu encode_gender=True: mã hóa Gender thành cột nhị phân [0, 1] và ghép nối ngang (np.hstack).
        4. Trả về ma trận numpy 2 chiều sẵn sàng cho thuật toán phân cụm.
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Ý LÀM:
        Thực hiện đồng thời fit(df) sau đó transform(df) trên tập Train.
        """
        pass


def compare_scalers(df: pd.DataFrame, feature_cols: List[str] = NUMERICAL_FEATURES) -> pd.DataFrame:
    """
    Ý LÀM:
    1. Áp dụng cả 2 phương pháp: StandardScaler và MinMaxScaler lên cùng tập dữ liệu.
    2. Tính toán các chỉ số thống kê (Min, Max, Mean, Std) trước và sau khi chuẩn hóa của từng biến.
    3. Tổng hợp thành DataFrame đối chiếu để người đọc dễ dàng so sánh hiệu quả của từng phương pháp.
    """
    pass

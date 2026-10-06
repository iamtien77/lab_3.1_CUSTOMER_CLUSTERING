"""
=============================================================================
MODULE: TIỀN XỬ LÝ DỮ LIỆU CƠ BẢN (DATA PREPROCESSING & CLEANING)
=============================================================================
Mục đích:
- Làm sạch dữ liệu cơ bản, xử lý kiểu dữ liệu (data types).
- Kiểm tra chi tiết giá trị thiếu (missing values) và giá trị ngoại lai (outliers)
  theo yêu cầu Bước 1 trong đề bài Practice Exercise 3.
=============================================================================
"""

from typing import Dict, Any, List
import pandas as pd
from src.config import NUMERICAL_FEATURES


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ý LÀM:
    1. Tạo một bản sao độc lập của DataFrame (df.copy()) để không ảnh hưởng dữ liệu gốc.
    2. Chuyển đổi các cột số (Age, Annual Income, Spending Score) về kiểu numeric (pd.to_numeric).
    3. Nếu có giá trị khuyết (missing values), xử lý bằng cách:
       - Điền trung vị (Median) cho các cột số.
       - Điền mode (giá trị xuất hiện nhiều nhất) cho cột phân loại như Gender.
    4. Trả về DataFrame đã được làm sạch.
    """
    pass


def audit_missing_and_outliers(df: pd.DataFrame, numerical_cols: List[str] = NUMERICAL_FEATURES) -> pd.DataFrame:
    """
    Ý LÀM (Đáp ứng Bước 1 trong đề bài - Data Exploration):
    1. Duyệt qua từng cột số trong numerical_cols.
    2. Đếm số lượng và tỷ lệ giá trị thiếu (Null count & Null percentage).
    3. Phát hiện Outlier bằng phương pháp Khoảng tứ phân vị (IQR Method - Tukey's Fences):
       - Q1 = quantile(0.25), Q3 = quantile(0.75), IQR = Q3 - Q1.
       - Ngưỡng dưới = Q1 - 1.5 * IQR.
       - Ngưỡng trên = Q3 + 1.5 * IQR.
       - Đếm các điểm nằm ngoài [Ngưỡng dưới, Ngưỡng trên].
    4. Tổng hợp thành bảng DataFrame báo cáo gồm các cột:
       [Đặc trưng, Null Count, Null %, Q1, Q3, IQR, Ngưỡng dưới, Ngưỡng trên, Số lượng Outlier, Tỷ lệ Outlier %].
    5. Trả về DataFrame báo cáo.
    """
    pass

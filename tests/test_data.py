"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: MODULE DỮ LIỆU (TEST DATA LOADER & CLEANING)
=============================================================================
"""

import unittest
from pathlib import Path
import pandas as pd
from src.config import DEFAULT_DATA_PATH


class TestDataLoader(unittest.TestCase):
    """
    Ý LÀM:
    1. test_file_exists: Kiểm tra file dữ liệu data/raw/Mall_Customers.csv có tồn tại không.
    2. test_schema_columns: Kiểm tra dữ liệu nạp lên có đủ các cột: CustomerID, Gender, Age, Annual Income, Spending Score.
    3. test_no_null_values: Kiểm tra tập dữ liệu không bị thiếu dữ liệu.
    4. test_unique_customer_ids: Kiểm tra CustomerID không bị lặp.
    5. test_split_ratio: Kiểm tra hàm split_data chia đúng tỷ lệ 80% Train và 20% Test.
    """
    pass


if __name__ == "__main__":
    unittest.main()

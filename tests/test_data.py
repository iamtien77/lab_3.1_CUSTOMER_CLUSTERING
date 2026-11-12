"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: MODULE DỮ LIỆU (TEST DATA LOADER & CLEANING)
=============================================================================
"""

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.config import DEFAULT_DATA_PATH, ID_COLUMN, SPENDING_COMPONENT_COLUMNS
from src.data.loader import load_raw_data, split_data, validate_schema
from src.data.preprocessing import audit_missing_and_outliers, clean_data


def _tao_dong_khach_hang_mau(
    ma_kh: int,
    nam_sinh: int,
    thu_nhap: float | None,
    chi_ruou: int = 100,
    chi_trai_cay: int = 50,
) -> dict:
    """Tạo 1 bản ghi khách hàng tổng hợp bám sát schema marketing_campaign.csv."""
    ban_ghi = {
        ID_COLUMN: ma_kh,
        "Year_Birth": nam_sinh,
        "Education": "Graduation",
        "Marital_Status": "Single",
        "Income": thu_nhap,
        "Kidhome": 0,
        "Teenhome": 0,
        "Dt_Customer": "01-01-2014",
        "Recency": 10,
        "MntWines": chi_ruou,
        "MntFruits": chi_trai_cay,
        "MntMeatProducts": 0,
        "MntFishProducts": 0,
        "MntSweetProducts": 0,
        "MntGoldProds": 0,
    }
    return ban_ghi


class TestDataLoader(unittest.TestCase):
    """Kiểm thử tại seam công khai cho module nạp, kiểm định, làm sạch và chia dữ liệu."""

    def test_file_exists_and_load_raw_data_tsv(self):
        """Kiểm tra tệp marketing_campaign.csv tồn tại và nạp đúng 2.240 dòng x 29 cột."""
        self.assertTrue(DEFAULT_DATA_PATH.exists(), f"Không tìm thấy tệp dữ liệu: {DEFAULT_DATA_PATH}")
        df = load_raw_data()
        self.assertEqual(df.shape, (2240, 29))
        self.assertIn(ID_COLUMN, df.columns)
        for col in SPENDING_COMPONENT_COLUMNS:
            self.assertIn(col, df.columns)

        with self.assertRaises(FileNotFoundError):
            load_raw_data(Path("duong_dan_khong_ton_tai.csv"))

    def test_validate_schema_raw_and_invalid_cases(self):
        """Kiểm tra validate_schema phát hiện đúng hợp lệ trên tập thật và lỗi trên tập giả lập."""
        df_raw = load_raw_data()
        ket_qua_that = validate_schema(df_raw)
        self.assertTrue(ket_qua_that["is_valid"])
        self.assertEqual(ket_qua_that["n_rows"], 2240)
        self.assertEqual(ket_qua_that["n_cols"], 29)
        self.assertEqual(ket_qua_that["duplicate_ids"], 0)
        self.assertEqual(ket_qua_that["total_nulls"], 24)

        df_thieu_cot = df_raw.drop(columns=["Income"])
        self.assertFalse(validate_schema(df_thieu_cot)["is_valid"])

        df_trung_id = pd.DataFrame([
            _tao_dong_khach_hang_mau(1, 1980, 40000.0),
            _tao_dong_khach_hang_mau(1, 1985, 50000.0),
        ])
        ket_qua_trung = validate_schema(df_trung_id)
        self.assertFalse(ket_qua_trung["is_valid"])
        self.assertEqual(ket_qua_trung["duplicate_ids"], 1)

    def test_clean_data_derives_features_filters_outliers_and_fills_nulls(self):
        """Kiểm tra tạo Age=2014-Year_Birth, Total_Spending=sum(Mnt*), lọc bất hợp lý và điền median."""
        df_gia = pd.DataFrame([
            _tao_dong_khach_hang_mau(1, 1893, 60000.0, chi_ruou=10, chi_trai_cay=10),   # Lỗi năm sinh < 1920
            _tao_dong_khach_hang_mau(2, 1977, 666666.0, chi_ruou=10, chi_trai_cay=10),  # Outlier thu nhập phi lý
            _tao_dong_khach_hang_mau(3, 1984, 30000.0, chi_ruou=100, chi_trai_cay=50),  # Hợp lệ -> Age=30, Total=150
            _tao_dong_khach_hang_mau(4, 1974, 50000.0, chi_ruou=200, chi_trai_cay=100), # Hợp lệ -> Age=40, Total=300
            _tao_dong_khach_hang_mau(5, 1994, None, chi_ruou=20, chi_trai_cay=5),       # Hợp lệ, thiếu Income -> median(30k, 50k)=40k
        ])

        df_sach = clean_data(df_gia)
        self.assertEqual(len(df_sach), 3)
        self.assertListEqual(df_sach[ID_COLUMN].tolist(), [3, 4, 5])
        self.assertEqual(df_sach.loc[df_sach[ID_COLUMN] == 3, "Age"].iloc[0], 30)
        self.assertEqual(df_sach.loc[df_sach[ID_COLUMN] == 3, "Total_Spending"].iloc[0], 150)
        self.assertAlmostEqual(df_sach.loc[df_sach[ID_COLUMN] == 5, "Income"].iloc[0], 40000.0)
        self.assertEqual(int(df_sach[["Age", "Income", "Total_Spending"]].isnull().sum().sum()), 0)

    def test_audit_missing_and_outliers_worked_example(self):
        """Kiểm tra tính toán Q1, Q3, IQR và đếm outlier khớp giá trị tính tay."""
        gia_tri = [10.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 100.0, None]
        df_test = pd.DataFrame({"Income": gia_tri})
        bao_cao = audit_missing_and_outliers(df_test, numerical_cols=["Income"])

        self.assertEqual(len(bao_cao), 1)
        dong = bao_cao.iloc[0]
        self.assertEqual(dong["Null Count"], 1)
        self.assertAlmostEqual(dong["Q1"], 22.0)
        self.assertAlmostEqual(dong["Q3"], 30.0)
        self.assertAlmostEqual(dong["IQR"], 8.0)
        self.assertAlmostEqual(dong["Ngưỡng dưới"], 10.0)
        self.assertAlmostEqual(dong["Ngưỡng trên"], 42.0)
        self.assertEqual(dong["Số lượng Outlier"], 1)

    def test_split_data_ratio_and_reproducibility(self):
        """Kiểm tra chia đúng tỷ lệ 80/20, lưu tệp vào thư mục tạm và tái lập theo seed."""
        df_100 = pd.DataFrame([
            _tao_dong_khach_hang_mau(i, 1980, 40000.0 + i * 100) for i in range(1, 101)
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            train_1, test_1 = split_data(df_100, test_size=0.20, random_state=42, output_dir=out_dir)
            train_2, test_2 = split_data(df_100, test_size=0.20, random_state=42, output_dir=out_dir)

            self.assertEqual(len(train_1), 80)
            self.assertEqual(len(test_1), 20)
            self.assertListEqual(train_1[ID_COLUMN].tolist(), train_2[ID_COLUMN].tolist())
            self.assertListEqual(test_1[ID_COLUMN].tolist(), test_2[ID_COLUMN].tolist())
            self.assertTrue((out_dir / "train.csv").exists())
            self.assertTrue((out_dir / "test.csv").exists())
            self.assertTrue((out_dir / "full_customers.csv").exists())

    def test_validate_schema_rejects_missing_id(self):
        """Kiểm tra validate_schema phát hiện được bản ghi thiếu ID (NaN)."""
        df_thieu_id = pd.DataFrame([
            _tao_dong_khach_hang_mau(1, 1980, 40000.0),
            _tao_dong_khach_hang_mau(None, 1985, 50000.0),
        ])
        ket_qua = validate_schema(df_thieu_id)
        self.assertFalse(ket_qua["is_valid"])
        self.assertEqual(ket_qua["missing_ids"], 1)

    def test_clean_data_rejects_future_birth_negative_spend_and_all_nan(self):
        """Kiểm tra filter_invalid_records chặn Year_Birth > 2014, Mnt* âm và clean_data chặn cột toàn NaN."""
        df_loi = pd.DataFrame([
            _tao_dong_khach_hang_mau(1, 2030, 40000.0, chi_ruou=50),
            _tao_dong_khach_hang_mau(2, 1985, 45000.0, chi_ruou=-10),
            _tao_dong_khach_hang_mau(3, 1984, 50000.0, chi_ruou=100),
        ])
        df_sach = clean_data(df_loi)
        self.assertListEqual(df_sach[ID_COLUMN].tolist(), [3])

        df_toan_nan = pd.DataFrame([
            _tao_dong_khach_hang_mau(10, 1980, None),
            _tao_dong_khach_hang_mau(11, 1982, None),
        ])
        with self.assertRaises(ValueError):
            clean_data(df_toan_nan)


if __name__ == "__main__":
    unittest.main()

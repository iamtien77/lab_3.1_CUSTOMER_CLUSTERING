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


def _build_sample_customer_row(
    customer_id: int | None,
    year_birth: int,
    income: float | None,
    wine_spend: int = 100,
    fruit_spend: int = 50,
) -> dict:
    """Tạo 1 bản ghi khách hàng tổng hợp bám sát schema marketing_campaign.csv."""
    record = {
        ID_COLUMN: customer_id,
        "Year_Birth": year_birth,
        "Education": "Graduation",
        "Marital_Status": "Single",
        "Income": income,
        "Kidhome": 0,
        "Teenhome": 0,
        "Dt_Customer": "01-01-2014",
        "Recency": 10,
        "MntWines": wine_spend,
        "MntFruits": fruit_spend,
        "MntMeatProducts": 0,
        "MntFishProducts": 0,
        "MntSweetProducts": 0,
        "MntGoldProds": 0,
    }
    return record


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
        raw_df = load_raw_data()
        real_schema_result = validate_schema(raw_df)
        self.assertTrue(real_schema_result["is_valid"])
        self.assertEqual(real_schema_result["n_rows"], 2240)
        self.assertEqual(real_schema_result["n_cols"], 29)
        self.assertEqual(real_schema_result["duplicate_ids"], 0)
        self.assertEqual(real_schema_result["total_nulls"], 24)

        missing_col_df = raw_df.drop(columns=["Income"])
        self.assertFalse(validate_schema(missing_col_df)["is_valid"])

        duplicate_id_df = pd.DataFrame([
            _build_sample_customer_row(1, 1980, 40000.0),
            _build_sample_customer_row(1, 1985, 50000.0),
        ])
        dup_schema_result = validate_schema(duplicate_id_df)
        self.assertFalse(dup_schema_result["is_valid"])
        self.assertEqual(dup_schema_result["duplicate_ids"], 1)

    def test_clean_data_derives_features_filters_outliers_and_fills_nulls(self):
        """Kiểm tra tạo Age=2014-Year_Birth, Total_Spending=sum(Mnt*), lọc bất hợp lý và điền median."""
        mock_df = pd.DataFrame([
            _build_sample_customer_row(1, 1893, 60000.0, wine_spend=10, fruit_spend=10),   # Lỗi năm sinh < 1920
            _build_sample_customer_row(2, 1977, 666666.0, wine_spend=10, fruit_spend=10),  # Outlier thu nhập phi lý
            _build_sample_customer_row(3, 1984, 30000.0, wine_spend=100, fruit_spend=50),  # Hợp lệ -> Age=30, Total=150
            _build_sample_customer_row(4, 1974, 50000.0, wine_spend=200, fruit_spend=100), # Hợp lệ -> Age=40, Total=300
            _build_sample_customer_row(5, 1994, None, wine_spend=20, fruit_spend=5),       # Hợp lệ, thiếu Income -> median(30k, 50k)=40k
        ])

        cleaned_df = clean_data(mock_df)
        self.assertEqual(len(cleaned_df), 3)
        self.assertListEqual(cleaned_df[ID_COLUMN].tolist(), [3, 4, 5])
        self.assertEqual(cleaned_df.loc[cleaned_df[ID_COLUMN] == 3, "Age"].iloc[0], 30)
        self.assertEqual(cleaned_df.loc[cleaned_df[ID_COLUMN] == 3, "Total_Spending"].iloc[0], 150)
        self.assertAlmostEqual(cleaned_df.loc[cleaned_df[ID_COLUMN] == 5, "Income"].iloc[0], 40000.0)
        self.assertEqual(int(cleaned_df[["Age", "Income", "Total_Spending"]].isnull().sum().sum()), 0)

    def test_audit_missing_and_outliers_worked_example(self):
        """Kiểm tra tính toán Q1, Q3, IQR và đếm outlier khớp giá trị tính tay."""
        sample_values = [10.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 100.0, None]
        df_test = pd.DataFrame({"Income": sample_values})
        audit_df = audit_missing_and_outliers(df_test, numerical_cols=["Income"])

        self.assertEqual(len(audit_df), 1)
        audit_row = audit_df.iloc[0]
        self.assertEqual(audit_row["Null Count"], 1)
        self.assertAlmostEqual(audit_row["Q1"], 22.0)
        self.assertAlmostEqual(audit_row["Q3"], 30.0)
        self.assertAlmostEqual(audit_row["IQR"], 8.0)
        self.assertAlmostEqual(audit_row["Ngưỡng dưới"], 10.0)
        self.assertAlmostEqual(audit_row["Ngưỡng trên"], 42.0)
        self.assertEqual(audit_row["Số lượng Outlier"], 1)

    def test_split_data_ratio_and_reproducibility(self):
        """Kiểm tra chia đúng tỷ lệ 80/20, lưu tệp vào thư mục tạm và tái lập theo seed."""
        df_100 = pd.DataFrame([
            _build_sample_customer_row(i, 1980, 40000.0 + i * 100) for i in range(1, 101)
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
        missing_id_df = pd.DataFrame([
            _build_sample_customer_row(1, 1980, 40000.0),
            _build_sample_customer_row(None, 1985, 50000.0),
        ])
        schema_result = validate_schema(missing_id_df)
        self.assertFalse(schema_result["is_valid"])
        self.assertEqual(schema_result["missing_ids"], 1)

    def test_clean_data_rejects_future_birth_negative_spend_and_all_nan(self):
        """Kiểm tra filter_invalid_records chặn Year_Birth > 2014, Mnt* âm và clean_data chặn cột toàn NaN."""
        invalid_df = pd.DataFrame([
            _build_sample_customer_row(1, 2030, 40000.0, wine_spend=50),
            _build_sample_customer_row(2, 1985, 45000.0, wine_spend=-10),
            _build_sample_customer_row(3, 1984, 50000.0, wine_spend=100),
        ])
        cleaned_df = clean_data(invalid_df)
        self.assertListEqual(cleaned_df[ID_COLUMN].tolist(), [3])

        all_nan_df = pd.DataFrame([
            _build_sample_customer_row(10, 1980, None),
            _build_sample_customer_row(11, 1982, None),
        ])
        with self.assertRaises(ValueError):
            clean_data(all_nan_df)


if __name__ == "__main__":
    unittest.main()

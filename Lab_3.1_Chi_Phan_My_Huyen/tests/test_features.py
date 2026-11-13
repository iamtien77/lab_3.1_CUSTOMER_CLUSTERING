"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: MODULE CHUẨN HÓA ĐẶC TRƯNG (TEST FEATURE SCALING)
=============================================================================
"""

import unittest
import numpy as np
import pandas as pd

from src.features.scaling import (
    FeatureScaler,
    compare_scalers,
    evaluate_scalers_on_clustering,
)


class TestFeatureScaling(unittest.TestCase):
    """Kiểm thử tại seam công khai cho chuẩn hóa đặc trưng và phòng chống rò rỉ dữ liệu."""

    def test_standard_scaler_worked_values_and_inverse(self):
        """Kiểm tra StandardScaler chuẩn hóa đúng giá trị tính tay và khôi phục bằng inverse_transform."""
        df_train = pd.DataFrame({
            "ID": [1, 2, 3],
            "Age": [20.0, 30.0, 40.0],
            "Income": [10000.0, 20000.0, 30000.0],
            "Total_Spending": [100.0, 200.0, 300.0],
        })
        scaler = FeatureScaler(method="standard")
        scaled_matrix = scaler.fit_transform(df_train)

        self.assertEqual(scaled_matrix.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(scaled_matrix)))
        self.assertTrue(np.allclose(scaled_matrix.mean(axis=0), 0.0, atol=1e-7))
        self.assertTrue(np.allclose(scaled_matrix.std(axis=0), 1.0, atol=1e-7))
        # Với [10, 20, 30], điểm đầu là -sqrt(3/2), giữa là 0, cuối là +sqrt(3/2)
        expected_value = np.sqrt(1.5)
        self.assertAlmostEqual(scaled_matrix[0, 0], -expected_value, places=6)
        self.assertAlmostEqual(scaled_matrix[1, 0], 0.0, places=6)
        self.assertAlmostEqual(scaled_matrix[2, 0], expected_value, places=6)

        raw_matrix = scaler.inverse_transform(scaled_matrix)
        self.assertTrue(np.allclose(raw_matrix, df_train[["Age", "Income", "Total_Spending"]].values))

    def test_minmax_scaler_worked_range(self):
        """Kiểm tra MinMaxScaler biến đổi [10, 20, 50] thành [0.0, 0.25, 1.0]."""
        df = pd.DataFrame({
            "Age": [10.0, 20.0, 50.0],
            "Income": [100.0, 300.0, 500.0],
            "Total_Spending": [0.0, 50.0, 100.0],
        })
        scaler = FeatureScaler(method="minmax")
        scaled_matrix = scaler.fit_transform(df)

        self.assertTrue(np.all(scaled_matrix >= -1e-9) and np.all(scaled_matrix <= 1.0 + 1e-9))
        self.assertAlmostEqual(scaled_matrix[0, 0], 0.0, places=6)
        self.assertAlmostEqual(scaled_matrix[1, 0], 0.25, places=6)
        self.assertAlmostEqual(scaled_matrix[2, 0], 1.0, places=6)

    def test_no_data_leakage_from_test_set(self):
        """Kiểm tra transform tập Test sử dụng thống kê đã fit trên Train (không tự fit lại)."""
        df_train = pd.DataFrame({"Age": [20.0, 30.0, 40.0], "Income": [10.0, 20.0, 30.0], "Total_Spending": [1.0, 2.0, 3.0]})
        df_test = pd.DataFrame({"Age": [50.0], "Income": [40.0], "Total_Spending": [4.0]})

        scaler = FeatureScaler(method="standard").fit(df_train)
        test_scaled_matrix = scaler.transform(df_test)

        # (50 - 30) / std([20,30,40]) = 20 / (10*sqrt(2/3)) = sqrt(6) ≈ 2.4494897
        self.assertAlmostEqual(test_scaled_matrix[0, 0], np.sqrt(6.0), places=6)

    def test_validation_guards_errors(self):
        """Kiểm tra cảnh báo rõ ràng khi chưa fit, chứa NaN, phương sai bằng 0 hoặc sai method."""
        with self.assertRaises(ValueError):
            FeatureScaler(method="invalid_method")

        scaler = FeatureScaler(method="standard")
        valid_df = pd.DataFrame({"Age": [20.0, 30.0], "Income": [10.0, 20.0], "Total_Spending": [5.0, 10.0]})
        with self.assertRaises(RuntimeError):
            scaler.transform(valid_df)

        nan_df = pd.DataFrame({"Age": [20.0, np.nan], "Income": [10.0, 20.0], "Total_Spending": [5.0, 10.0]})
        with self.assertRaises(ValueError):
            scaler.fit(nan_df)

        constant_df = pd.DataFrame({"Age": [25.0, 25.0], "Income": [10.0, 20.0], "Total_Spending": [5.0, 10.0]})
        with self.assertRaises(ValueError):
            scaler.fit(constant_df)

    def test_compare_scalers_summary_table(self):
        """Kiểm tra hàm compare_scalers xuất đúng bảng so sánh 4 phương pháp x 3 đặc trưng."""
        df = pd.DataFrame({
            "Age": [20.0, 30.0, 40.0, 50.0],
            "Income": [20000.0, 40000.0, 60000.0, 80000.0],
            "Total_Spending": [100.0, 300.0, 500.0, 700.0],
        })
        comparison_df = compare_scalers(df)
        self.assertEqual(len(comparison_df), 12)
        for col_name in ("Median", "IQR", "Skew"):
            self.assertIn(col_name, comparison_df.columns)
        minmax_rows = comparison_df[comparison_df["Phương pháp"] == "MinMaxScaler"]
        self.assertTrue(np.allclose(minmax_rows["Min"], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(minmax_rows["Max"], 1.0, atol=1e-6))
        robust_rows = comparison_df[comparison_df["Phương pháp"] == "RobustScaler"]
        self.assertTrue(np.allclose(robust_rows["Median"], 0.0, atol=1e-6))

    def test_learned_params_verification_and_clustering_comparison(self):
        """Kiểm tra trích xuất tham số Train, hậu kiểm Train/Test và đánh giá 3 scaler trên K-Means."""
        df_train = pd.DataFrame({
            "Age": [20.0, 22.0, 24.0, 50.0, 54.0, 58.0, 35.0, 42.0],
            "Income": [18000.0, 20000.0, 22000.0, 85000.0, 90000.0, 95000.0, 48000.0, 62000.0],
            "Total_Spending": [60.0, 80.0, 90.0, 1500.0, 1700.0, 1900.0, 450.0, 900.0],
        })
        df_test = pd.DataFrame({
            "Age": [18.0, 65.0],
            "Income": [15000.0, 120000.0],
            "Total_Spending": [50.0, 2200.0],
        })

        scaler = FeatureScaler(method="standard").fit(df_train)
        params_df = scaler.get_learned_params()
        self.assertEqual(len(params_df), 3)
        self.assertIn("Tham số tâm", params_df.columns)

        train_matrix = scaler.transform(df_train)
        verify_train = scaler.verify_scaled_matrix(train_matrix, is_train=True)
        self.assertTrue(bool(verify_train["Đạt chuẩn"].all()))

        minmax = FeatureScaler(method="minmax").fit(df_train)
        verify_minmax_test = minmax.verify_scaled_matrix(minmax.transform(df_test), is_train=False)
        self.assertTrue(all("Ngoài [0, 1]: 2 mẫu" in text for text in verify_minmax_test["Hậu kiểm"]))

        with self.assertRaises(ValueError):
            scaler.inverse_transform(np.array([1.0, 2.0, 3.0]))

        eval_df = evaluate_scalers_on_clustering(df_train, test_df=df_test, fixed_k=2, k_range=[2, 3])
        self.assertListEqual(eval_df["Scaler"].tolist(), ["standard", "minmax", "robust"])
        self.assertIn("Silhouette Train", eval_df.columns)
        self.assertIn("Silhouette Test", eval_df.columns)


if __name__ == "__main__":
    unittest.main()

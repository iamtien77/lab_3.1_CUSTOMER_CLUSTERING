"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: MODULE DỰ ĐOÁN TRIỂN KHAI (TEST DEPLOYMENT PREDICTOR)
=============================================================================
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from src.deployment.predictor import CustomerSegmentPredictor
from src.evaluation.profiling import create_cluster_profiles, generate_customer_personas
from src.features.scaling import FeatureScaler
from src.models.kmeans import KMeansTrainer
from src.models.kmedoids import KMedoidsTrainer
from src.models.persistence import save_model


class TestPredictor(unittest.TestCase):
    """Kiểm thử tại seam công khai cho suy luận phân khúc khách hàng mới."""

    def _tao_goi_mo_hinh_tam(self, tmp_dir: Path) -> Path:
        """Huấn luyện nhanh mô hình K=2 và lưu vào thư mục tạm phục vụ test suy luận."""
        df_train = pd.DataFrame({
            "Age": [24.0, 26.0, 28.0, 52.0, 55.0, 58.0],
            "Income": [22000.0, 24000.0, 25000.0, 88000.0, 92000.0, 95000.0],
            "Total_Spending": [80.0, 100.0, 120.0, 1600.0, 1750.0, 1900.0],
        })
        scaler = FeatureScaler(method="standard")
        ma_tran_scaled = scaler.fit_transform(df_train)
        model = KMeansTrainer(random_state=42).fit_final_model(ma_tran_scaled, optimal_k=2)

        profile_df = create_cluster_profiles(df_train, model.labels_, output_path=tmp_dir / "profiles.csv")
        personas = generate_customer_personas(profile_df)
        personas_map = {p["cluster_id"]: p for p in personas}
        trung_vi_mau = {
            "Age": float(df_train["Age"].median()),
            "Income": float(df_train["Income"].median()),
            "Total_Spending": float(df_train["Total_Spending"].median()),
        }

        return save_model(
            model_object=model,
            scaler_object=scaler,
            model_name="pipeline_test.pkl",
            metadata={"n_clusters": 2, "personas": personas_map, "feature_medians": trung_vi_mau},
            output_dir=tmp_dir,
        )

    def test_predict_single_customer_and_validation(self):
        """Kiểm tra dự đoán 1 khách hàng trả đủ trường và từ chối giá trị âm bất hợp lý."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._tao_goi_mo_hinh_tam(tmp_dir)
            predictor = CustomerSegmentPredictor(model_filename="pipeline_test.pkl", model_dir=tmp_dir)

            ket_qua = predictor.predict_single_customer(age=54, annual_income=91000.0, total_spending=1700.0)
            for khoa in ["customer_input", "assigned_cluster", "persona_name", "marketing_recommendation", "distance_to_center"]:
                self.assertIn(khoa, ket_qua)
            self.assertIn(ket_qua["assigned_cluster"], {0, 1})
            self.assertIn("VIP", ket_qua["persona_name"])
            self.assertGreaterEqual(ket_qua["distance_to_center"], 0.0)

            with self.assertRaises(ValueError):
                predictor.predict_single_customer(age=-5, annual_income=50000.0, total_spending=300.0)

    def test_predict_batch_shape_and_valid_cluster_range(self):
        """Kiểm tra dự đoán hàng loạt bảo toàn số dòng và gán nhãn đúng phạm vi cụm."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._tao_goi_mo_hinh_tam(tmp_dir)
            predictor = CustomerSegmentPredictor(model_filename="pipeline_test.pkl", model_dir=tmp_dir)

            df_moi = pd.DataFrame({
                "Age": [25.0, 56.0],
                "Income": [23000.0, 93000.0],
                "Total_Spending": [95.0, 1800.0],
            })
            ket_qua_df = predictor.predict_batch(df_moi)
            self.assertEqual(len(ket_qua_df), 2)
            self.assertIn("Predicted_Cluster", ket_qua_df.columns)
            self.assertIn("Persona_Name", ket_qua_df.columns)
            self.assertTrue(set(ket_qua_df["Predicted_Cluster"]).issubset({0, 1}))
            self.assertNotEqual(
                ket_qua_df.loc[0, "Predicted_Cluster"],
                ket_qua_df.loc[1, "Predicted_Cluster"],
            )

    def test_predict_batch_handles_nan_income_with_train_medians(self):
        """Kiểm tra predict_batch tự điền khuyết Income bằng feature_medians đã lưu, không văng lỗi NaN."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._tao_goi_mo_hinh_tam(tmp_dir)
            predictor = CustomerSegmentPredictor(model_filename="pipeline_test.pkl", model_dir=tmp_dir)

            df_thieu = pd.DataFrame({
                "Age": [54.0, 25.0],
                "Income": [np.nan, 23000.0],
                "Total_Spending": [1700.0, 95.0],
            })
            ket_qua_df = predictor.predict_batch(df_thieu)
            self.assertEqual(len(ket_qua_df), 2)
            self.assertTrue(np.isfinite(ket_qua_df["Income"]).all())
            self.assertTrue(set(ket_qua_df["Predicted_Cluster"]).issubset({0, 1}))

    def test_cli_predictor_runs_against_saved_model(self):
        """Kiểm tra giao diện CLI predictor chạy độc lập và trả mã thoát 0 trên mô hình đã lưu."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._tao_goi_mo_hinh_tam(tmp_dir)
            project_root = Path(__file__).resolve().parents[1]
            ket_qua_cmd = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.deployment.predictor",
                    "--age",
                    "54",
                    "--income",
                    "91000",
                    "--spending",
                    "1700",
                    "--model",
                    "pipeline_test.pkl",
                    "--model-dir",
                    str(tmp_dir),
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(ket_qua_cmd.returncode, 0, msg=ket_qua_cmd.stderr)
            self.assertIn("Cụm dự đoán", ket_qua_cmd.stdout)
            self.assertIn("VIP", ket_qua_cmd.stdout)

    def test_predictor_rejects_nan_inf_non_predict_and_supports_index_and_manhattan(self):
        """Kiểm tra chặn NaN/Inf, mô hình thiếu predict(), giữ custom index và khoảng cách L1 cho K-Medoids."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._tao_goi_mo_hinh_tam(tmp_dir)
            predictor = CustomerSegmentPredictor(model_filename="pipeline_test.pkl", model_dir=tmp_dir)

            with self.assertRaises(ValueError):
                predictor.predict_single_customer(age=float("nan"), annual_income=50000.0, total_spending=200.0)
            with self.assertRaises(ValueError):
                predictor.predict_single_customer(age=35.0, annual_income=float("inf"), total_spending=200.0)

            df_custom_idx = pd.DataFrame(
                {
                    "Age": [25.0, 56.0],
                    "Income": [23000.0, 93000.0],
                    "Total_Spending": [95.0, 1800.0],
                },
                index=["KH_A", "KH_B"],
            )
            ket_qua_idx = predictor.predict_batch(df_custom_idx)
            self.assertListEqual(ket_qua_idx.index.tolist(), ["KH_A", "KH_B"])

            # Mô hình AgglomerativeClustering không có predict() phải bị chặn ngay khi khởi tạo
            agg = AgglomerativeClustering(n_clusters=2).fit(np.array([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]]))
            save_model(agg, predictor.scaler, model_name="no_predict.pkl", metadata={}, output_dir=tmp_dir)
            with self.assertRaises(TypeError):
                CustomerSegmentPredictor(model_filename="no_predict.pkl", model_dir=tmp_dir)

            # Kiểm tra K-Medoids Manhattan dùng khoảng cách L1 (cityblock)
            df_km = pd.DataFrame({
                "Age": [24.0, 26.0, 52.0, 55.0],
                "Income": [22000.0, 24000.0, 88000.0, 92000.0],
                "Total_Spending": [80.0, 100.0, 1600.0, 1750.0],
            })
            scaler_km = FeatureScaler(method="standard")
            ma_tran_km = scaler_km.fit_transform(df_km)
            km_l1 = KMedoidsTrainer(n_clusters=2, metric="manhattan", random_state=42)
            km_l1.fit_predict(ma_tran_km)
            save_model(km_l1, scaler_km, model_name="kmedoids_l1.pkl", metadata={}, output_dir=tmp_dir)

            pred_l1 = CustomerSegmentPredictor(model_filename="kmedoids_l1.pkl", model_dir=tmp_dir)
            kq_l1 = pred_l1.predict_single_customer(age=54.0, annual_income=90000.0, total_spending=1700.0)
            x_test = scaler_km.transform(pd.DataFrame([{"Age": 54.0, "Income": 90000.0, "Total_Spending": 1700.0}]))
            tam_chon = km_l1.cluster_centers_[kq_l1["assigned_cluster"]]
            l1_ky_vong = round(float(np.sum(np.abs(x_test[0] - tam_chon))), 4)
            self.assertAlmostEqual(kq_l1["distance_to_center"], l1_ky_vong, places=4)


if __name__ == "__main__":
    unittest.main()

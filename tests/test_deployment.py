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

    def test_batch_rejects_invalid_domain_with_row_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._create_temp_model_bundle(Path(tmp))
            predictor = CustomerSegmentPredictor("pipeline_test.pkl", model_dir=tmp)
            for column, value in [("Age", -5), ("Age", 0), ("Age", 121),
                                  ("Income", -100), ("Total_Spending", -20),
                                  ("Income", np.inf), ("Income", "invalid")]:
                with self.subTest(column=column, value=value):
                    row = {"Age": 35, "Income": 50000, "Total_Spending": 300}
                    row[column] = value
                    with self.assertRaisesRegex(ValueError, "bad_row"):
                        predictor.predict_batch(pd.DataFrame([row], index=["bad_row"]))

    def test_batch_missing_income_requires_saved_train_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._create_temp_model_bundle(Path(tmp))
            predictor = CustomerSegmentPredictor("pipeline_test.pkl", model_dir=tmp)
            rows = pd.DataFrame({"Age": [35, 40], "Income": [np.nan, 10000],
                                 "Total_Spending": [500, 400]})
            for medians in [None, {}, {"Age": 35}, {"Income": np.nan}, {"Income": np.inf}]:
                with self.subTest(medians=medians):
                    predictor.metadata["feature_medians"] = medians
                    with self.assertRaisesRegex(ValueError, "Income"):
                        predictor.predict_batch(rows)

    def test_batch_rejects_negative_raw_component_even_when_total_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._create_temp_model_bundle(Path(tmp))
            predictor = CustomerSegmentPredictor("pipeline_test.pkl", model_dir=tmp)
            row = {"Year_Birth": 1980, "Income": 50000, "MntWines": -10,
                   "MntFruits": 100, "MntMeatProducts": 0, "MntFishProducts": 0,
                   "MntSweetProducts": 0, "MntGoldProds": 0}
            with self.assertRaisesRegex(ValueError, "bad_raw"):
                predictor.predict_batch(pd.DataFrame([row], index=["bad_raw"]))

    def test_batch_imputation_is_independent_and_complete_input_needs_no_medians(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._create_temp_model_bundle(Path(tmp))
            predictor = CustomerSegmentPredictor("pipeline_test.pkl", model_dir=tmp)
            rows = pd.DataFrame({"Age": [35, 40], "Income": [np.nan, 10000],
                                 "Total_Spending": [500, 400]}, index=["target", "other"])
            first = predictor.predict_batch(rows)
            rows.loc["other", "Income"] = 100000
            second = predictor.predict_batch(rows)
            self.assertEqual(first.loc["target", "Income"], 56500)
            self.assertEqual(first.loc["target", "Predicted_Cluster"], second.loc["target", "Predicted_Cluster"])
            self.assertListEqual(first.index.tolist(), rows.index.tolist())
            self.assertTrue(pd.isna(rows.loc["target", "Income"]))
            predictor.metadata.pop("feature_medians")
            complete = rows.loc[["other"]]
            batch = predictor.predict_batch(complete)
            single = predictor.predict_single_customer(40, 100000, 400)
            self.assertEqual(batch.iloc[0]["Predicted_Cluster"], single["assigned_cluster"])

    def test_batch_raw_missing_spend_uses_train_total_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._create_temp_model_bundle(Path(tmp))
            predictor = CustomerSegmentPredictor("pipeline_test.pkl", model_dir=tmp)
            row = {"Year_Birth": 1980, "Income": 50000, "MntWines": np.nan,
                   "MntFruits": 100, "MntMeatProducts": 0, "MntFishProducts": 0,
                   "MntSweetProducts": 0, "MntGoldProds": 0}
            result = predictor.predict_batch(pd.DataFrame([row]))
            self.assertEqual(result.iloc[0]["Age"], 34)
            self.assertEqual(result.iloc[0]["Total_Spending"], predictor.metadata["feature_medians"]["Total_Spending"])

    def _create_temp_model_bundle(self, tmp_dir: Path) -> Path:
        """Huấn luyện nhanh mô hình K=2 và lưu vào thư mục tạm phục vụ test suy luận."""
        df_train = pd.DataFrame({
            "Age": [24.0, 26.0, 28.0, 52.0, 55.0, 58.0],
            "Income": [22000.0, 24000.0, 25000.0, 88000.0, 92000.0, 95000.0],
            "Total_Spending": [80.0, 100.0, 120.0, 1600.0, 1750.0, 1900.0],
        })
        scaler = FeatureScaler(method="standard")
        scaled_matrix = scaler.fit_transform(df_train)
        model = KMeansTrainer(random_state=42).fit_final_model(scaled_matrix, optimal_k=2)

        profile_df = create_cluster_profiles(df_train, model.labels_, output_path=tmp_dir / "profiles.csv")
        personas = generate_customer_personas(profile_df)
        personas_map = {p["cluster_id"]: p for p in personas}
        sample_medians = {
            "Age": float(df_train["Age"].median()),
            "Income": float(df_train["Income"].median()),
            "Total_Spending": float(df_train["Total_Spending"].median()),
        }

        return save_model(
            model_object=model,
            scaler_object=scaler,
            model_name="pipeline_test.pkl",
            metadata={"n_clusters": 2, "personas": personas_map, "feature_medians": sample_medians},
            output_dir=tmp_dir,
        )

    def test_predict_single_customer_and_validation(self):
        """Kiểm tra dự đoán 1 khách hàng trả đủ trường và từ chối giá trị âm bất hợp lý."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._create_temp_model_bundle(tmp_dir)
            predictor = CustomerSegmentPredictor(model_filename="pipeline_test.pkl", model_dir=tmp_dir)

            prediction = predictor.predict_single_customer(age=54, annual_income=91000.0, total_spending=1700.0)
            for key in ["customer_input", "assigned_cluster", "persona_name", "marketing_recommendation", "distance_to_center"]:
                self.assertIn(key, prediction)
            self.assertIn(prediction["assigned_cluster"], {0, 1})
            self.assertIn("VIP", prediction["persona_name"])
            self.assertGreaterEqual(prediction["distance_to_center"], 0.0)

            with self.assertRaises(ValueError):
                predictor.predict_single_customer(age=-5, annual_income=50000.0, total_spending=300.0)

    def test_predict_batch_shape_and_valid_cluster_range(self):
        """Kiểm tra dự đoán hàng loạt bảo toàn số dòng và gán nhãn đúng phạm vi cụm."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._create_temp_model_bundle(tmp_dir)
            predictor = CustomerSegmentPredictor(model_filename="pipeline_test.pkl", model_dir=tmp_dir)

            new_df = pd.DataFrame({
                "Age": [25.0, 56.0],
                "Income": [23000.0, 93000.0],
                "Total_Spending": [95.0, 1800.0],
            })
            pred_df = predictor.predict_batch(new_df)
            self.assertEqual(len(pred_df), 2)
            self.assertIn("Predicted_Cluster", pred_df.columns)
            self.assertIn("Persona_Name", pred_df.columns)
            self.assertTrue(set(pred_df["Predicted_Cluster"]).issubset({0, 1}))
            self.assertNotEqual(
                pred_df.loc[0, "Predicted_Cluster"],
                pred_df.loc[1, "Predicted_Cluster"],
            )

    def test_predict_batch_handles_nan_income_with_train_medians(self):
        """Kiểm tra predict_batch tự điền khuyết Income bằng feature_medians đã lưu, không văng lỗi NaN."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._create_temp_model_bundle(tmp_dir)
            predictor = CustomerSegmentPredictor(model_filename="pipeline_test.pkl", model_dir=tmp_dir)

            missing_income_df = pd.DataFrame({
                "Age": [54.0, 25.0],
                "Income": [np.nan, 23000.0],
                "Total_Spending": [1700.0, 95.0],
            })
            pred_df = predictor.predict_batch(missing_income_df)
            self.assertEqual(len(pred_df), 2)
            self.assertTrue(np.isfinite(pred_df["Income"]).all())
            self.assertTrue(set(pred_df["Predicted_Cluster"]).issubset({0, 1}))

    def test_cli_predictor_runs_against_saved_model(self):
        """Kiểm tra giao diện CLI predictor chạy độc lập và trả mã thoát 0 trên mô hình đã lưu."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._create_temp_model_bundle(tmp_dir)
            project_root = Path(__file__).resolve().parents[1]
            cmd_result = subprocess.run(
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
            self.assertEqual(cmd_result.returncode, 0, msg=cmd_result.stderr)
            self.assertIn("Cụm dự đoán", cmd_result.stdout)
            self.assertIn("VIP", cmd_result.stdout)

    def test_predictor_rejects_nan_inf_non_predict_and_supports_index_and_manhattan(self):
        """Kiểm tra chặn NaN/Inf, mô hình thiếu predict(), giữ custom index và khoảng cách L1 cho K-Medoids."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self._create_temp_model_bundle(tmp_dir)
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
            idx_pred_df = predictor.predict_batch(df_custom_idx)
            self.assertListEqual(idx_pred_df.index.tolist(), ["KH_A", "KH_B"])

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
            km_matrix = scaler_km.fit_transform(df_km)
            kmedoids_l1 = KMedoidsTrainer(n_clusters=2, metric="manhattan", random_state=42)
            kmedoids_l1.fit_predict(km_matrix)
            save_model(kmedoids_l1, scaler_km, model_name="kmedoids_l1.pkl", metadata={}, output_dir=tmp_dir)

            predictor_l1 = CustomerSegmentPredictor(model_filename="kmedoids_l1.pkl", model_dir=tmp_dir)
            result_l1 = predictor_l1.predict_single_customer(age=54.0, annual_income=90000.0, total_spending=1700.0)
            scaled_test_point = scaler_km.transform(pd.DataFrame([{"Age": 54.0, "Income": 90000.0, "Total_Spending": 1700.0}]))
            selected_center = kmedoids_l1.cluster_centers_[result_l1["assigned_cluster"]]
            expected_l1_distance = round(float(np.sum(np.abs(scaled_test_point[0] - selected_center))), 4)
            self.assertAlmostEqual(result_l1["distance_to_center"], expected_l1_distance, places=4)


if __name__ == "__main__":
    unittest.main()

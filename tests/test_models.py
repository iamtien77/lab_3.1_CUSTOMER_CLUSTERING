"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: MODULE CÁC THUẬT TOÁN PHÂN CỤM (TEST CLUSTERING MODELS)
=============================================================================
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.features.scaling import FeatureScaler
from src.models.dbscan import DBSCANTrainer
from src.models.hierarchical import HierarchicalTrainer
from src.models.kmeans import KMeansTrainer
from src.models.kmedoids import KMedoidsTrainer
from src.models.persistence import load_model, save_model


def _tao_ba_cum_tach_biet() -> np.ndarray:
    """Tạo 15 điểm 2D gồm 3 cụm tách biệt rõ quanh (0,0), (10,0), (5,10)."""
    cum_0 = np.array([[0.0, 0.0], [0.2, 0.1], [-0.2, 0.1], [0.1, -0.2], [-0.1, -0.1]])
    cum_1 = np.array([[10.0, 0.0], [10.2, 0.1], [9.8, 0.1], [10.1, -0.2], [9.9, -0.1]])
    cum_2 = np.array([[5.0, 10.0], [5.2, 10.1], [4.8, 10.1], [5.1, 9.8], [4.9, 9.9]])
    return np.vstack([cum_0, cum_1, cum_2])


class TestClusteringModels(unittest.TestCase):
    """Kiểm thử tại seam công khai cho 4 thuật toán phân cụm và lưu trữ mô hình."""

    def test_kmeans_optimal_k_and_final_fit(self):
        """Kiểm tra K-Means phát hiện đúng K=3 tối ưu, Inertia giảm dần và có guard mẫu tối thiểu."""
        ma_tran = _tao_ba_cum_tach_biet()
        trainer = KMeansTrainer(random_state=42)

        bang_k = trainer.search_optimal_k(ma_tran, k_range=range(2, 6))
        self.assertEqual(list(bang_k["K"]), [2, 3, 4, 5])
        self.assertTrue(all(np.diff(bang_k["Inertia"]) <= 0))
        k_tot_nhat = int(bang_k.loc[bang_k["Silhouette"].idxmax(), "K"])
        self.assertEqual(k_tot_nhat, 3)

        model = trainer.fit_final_model(ma_tran, optimal_k=3)
        self.assertEqual(len(np.unique(model.labels_)), 3)
        self.assertEqual(model.cluster_centers_.shape, (3, 2))

        with self.assertRaises(ValueError):
            trainer.search_optimal_k(np.array([[1.0, 1.0], [2.0, 2.0]]), k_range=range(2, 4))

    def test_hierarchical_linkage_cophenetic_and_fit(self):
        """Kiểm tra ma trận liên kết (n-1, 4), ngưỡng cắt cây, hệ số Cophenetic > 0.7 và gán đủ 3 cụm."""
        ma_tran = _tao_ba_cum_tach_biet()
        trainer = HierarchicalTrainer(linkage_method="ward")
        with self.assertRaises(RuntimeError):
            trainer.compute_cut_threshold(3)

        ma_tran_lien_ket = trainer.compute_linkage_matrix(ma_tran)
        self.assertEqual(ma_tran_lien_ket.shape, (len(ma_tran) - 1, 4))
        self.assertGreater(trainer.cophenetic_corr_, 0.7)

        nguong_cat = trainer.compute_cut_threshold(3)
        self.assertGreater(nguong_cat, ma_tran_lien_ket[-3, 2])
        self.assertLess(nguong_cat, ma_tran_lien_ket[-2, 2])

        model = trainer.fit_model(ma_tran, n_clusters=3)
        self.assertEqual(len(model.labels_), len(ma_tran))
        self.assertEqual(len(np.unique(model.labels_)), 3)

    def test_dbscan_k_distances_grid_search_and_noise(self):
        """Kiểm tra đồ thị k-NN (loại tự khoảng cách 0), grid search và tách đúng điểm nhiễu nhãn -1."""
        cum_a = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, -0.1]])
        cum_b = np.array([[10.0, 10.0], [10.1, 10.1], [9.9, 10.1], [10.1, 9.9], [9.9, 9.9]])
        nhieu = np.array([[50.0, 50.0]])
        ma_tran = np.vstack([cum_a, cum_b, nhieu])

        trainer = DBSCANTrainer()
        k_dist_k1 = trainer.compute_k_distances(np.array([[0.0, 0.0], [3.0, 4.0], [20.0, 20.0]]), k=1)
        self.assertAlmostEqual(k_dist_k1[0], 5.0, places=6)
        self.assertAlmostEqual(k_dist_k1[1], 5.0, places=6)
        with self.assertRaises(ValueError):
            trainer.compute_k_distances(np.array([[0.0, 0.0], [1.0, 0.0]]), k=2)

        k_dist = trainer.compute_k_distances(ma_tran, k=3)
        self.assertEqual(len(k_dist), len(ma_tran))
        self.assertTrue(np.all(np.diff(k_dist) >= 0))
        self.assertGreater(k_dist[-1], 40.0)

        grid_df = trainer.grid_search_params(ma_tran, eps_values=[0.5, 1.0], min_samples_values=[3])
        self.assertEqual(len(grid_df), 2)
        self.assertTrue((grid_df["Số cụm"] == 2).all())
        self.assertTrue((grid_df["Số điểm nhiễu"] == 1).all())

        model = trainer.fit_model(ma_tran, eps=0.8, min_samples=3)
        self.assertEqual(model.labels_[-1], -1)
        self.assertEqual(len(set(model.labels_) - {-1}), 2)

    def test_kmedoids_centers_are_real_points_and_search_k(self):
        """Kiểm tra tâm cụm K-Medoids là điểm thật thuộc tập dữ liệu và tìm đúng K=2 cho 2 cụm."""
        ma_tran = np.array([
            [0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2],
            [10.0, 10.0], [10.2, 10.0], [10.0, 10.2], [10.2, 10.2],
        ])
        trainer = KMedoidsTrainer(n_clusters=2, random_state=42)
        labels = trainer.fit_predict(ma_tran)

        self.assertEqual(len(labels), 8)
        self.assertEqual(len(trainer.medoid_indices_), 2)
        for center in trainer.cluster_centers_:
            khop_diem_that = any(np.allclose(center, row) for row in ma_tran)
            self.assertTrue(khop_diem_that)
        self.assertEqual(len(set(labels[:4])), 1)
        self.assertEqual(len(set(labels[4:])), 1)
        self.assertNotEqual(labels[0], labels[4])

        bang_k = trainer.search_optimal_k(ma_tran, k_range=range(2, 5))
        self.assertEqual(int(bang_k.loc[bang_k["Silhouette"].idxmax(), "K"]), 2)

    def test_persistence_save_and_load_roundtrip(self):
        """Kiểm tra lưu gói (model + scaler + metadata) xuống đĩa và nạp lại khớp dự đoán."""
        df = pd.DataFrame({
            "Age": [22.0, 24.0, 23.0, 55.0, 58.0, 56.0],
            "Income": [20000.0, 22000.0, 21000.0, 90000.0, 95000.0, 92000.0],
            "Total_Spending": [50.0, 60.0, 55.0, 1500.0, 1600.0, 1550.0],
        })
        scaler = FeatureScaler(method="standard")
        ma_tran_scaled = scaler.fit_transform(df)
        model = KMeansTrainer(random_state=42).fit_final_model(ma_tran_scaled, optimal_k=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            saved_path = save_model(
                model_object=model,
                scaler_object=scaler,
                model_name="test_pipeline.pkl",
                metadata={"n_clusters": 2},
                output_dir=out_dir,
            )
            self.assertTrue(saved_path.exists())

            bundle = load_model("test_pipeline.pkl", input_dir=out_dir)
            self.assertIn("model", bundle)
            self.assertIn("scaler", bundle)
            self.assertEqual(bundle["metadata"]["n_clusters"], 2)

            nhan_tai = bundle["model"].predict(bundle["scaler"].transform(df))
            self.assertTrue(np.array_equal(model.labels_, nhan_tai))

            with self.assertRaises(FileNotFoundError):
                load_model("khong_ton_tai.pkl", input_dir=out_dir)

    def test_duplicate_points_and_nan_guards_kmeans_kmedoids(self):
        """Kiểm tra từ chối K > số điểm duy nhất ở K-Means / K-Medoids và NaN đầu vào K-Medoids."""
        ma_tran_trung = np.array([[0.0, 0.0]] * 5 + [[10.0, 10.0]] * 5)
        with self.assertRaises(ValueError):
            KMeansTrainer(random_state=42).fit_final_model(ma_tran_trung, optimal_k=3)
        with self.assertRaises(ValueError):
            KMedoidsTrainer(n_clusters=3, random_state=42).fit_predict(ma_tran_trung)

        ma_tran_nan = np.array([[0.0, np.nan], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        with self.assertRaises(ValueError):
            KMedoidsTrainer(n_clusters=2, random_state=42).fit_predict(ma_tran_nan)


if __name__ == "__main__":
    unittest.main()

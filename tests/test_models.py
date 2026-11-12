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


def _make_three_separated_clusters() -> np.ndarray:
    """Tạo 15 điểm 2D gồm 3 cụm tách biệt rõ quanh (0,0), (10,0), (5,10)."""
    cluster_0 = np.array([[0.0, 0.0], [0.2, 0.1], [-0.2, 0.1], [0.1, -0.2], [-0.1, -0.1]])
    cluster_1 = np.array([[10.0, 0.0], [10.2, 0.1], [9.8, 0.1], [10.1, -0.2], [9.9, -0.1]])
    cluster_2 = np.array([[5.0, 10.0], [5.2, 10.1], [4.8, 10.1], [5.1, 9.8], [4.9, 9.9]])
    return np.vstack([cluster_0, cluster_1, cluster_2])


class TestClusteringModels(unittest.TestCase):
    """Kiểm thử tại seam công khai cho 4 thuật toán phân cụm và lưu trữ mô hình."""

    def test_kmeans_optimal_k_and_final_fit(self):
        """Kiểm tra K-Means phát hiện đúng K=3 tối ưu, Inertia giảm dần và có guard mẫu tối thiểu."""
        matrix = _make_three_separated_clusters()
        trainer = KMeansTrainer(random_state=42)

        k_search_df = trainer.search_optimal_k(matrix, k_range=range(2, 6))
        self.assertEqual(list(k_search_df["K"]), [2, 3, 4, 5])
        self.assertTrue(all(np.diff(k_search_df["Inertia"]) <= 0))
        best_k = int(k_search_df.loc[k_search_df["Silhouette"].idxmax(), "K"])
        self.assertEqual(best_k, 3)

        model = trainer.fit_final_model(matrix, optimal_k=3)
        self.assertEqual(len(np.unique(model.labels_)), 3)
        self.assertEqual(model.cluster_centers_.shape, (3, 2))

        with self.assertRaises(ValueError):
            trainer.search_optimal_k(np.array([[1.0, 1.0], [2.0, 2.0]]), k_range=range(2, 4))

    def test_hierarchical_linkage_cophenetic_and_fit(self):
        """Kiểm tra ma trận liên kết (n-1, 4), ngưỡng cắt cây, hệ số Cophenetic > 0.7 và gán đủ 3 cụm."""
        matrix = _make_three_separated_clusters()
        trainer = HierarchicalTrainer(linkage_method="ward")
        with self.assertRaises(RuntimeError):
            trainer.compute_cut_threshold(3)

        linkage_matrix = trainer.compute_linkage_matrix(matrix)
        self.assertEqual(linkage_matrix.shape, (len(matrix) - 1, 4))
        self.assertGreater(trainer.cophenetic_corr_, 0.7)

        cut_threshold = trainer.compute_cut_threshold(3)
        self.assertGreater(cut_threshold, linkage_matrix[-3, 2])
        self.assertLess(cut_threshold, linkage_matrix[-2, 2])

        model = trainer.fit_model(matrix, n_clusters=3)
        self.assertEqual(len(model.labels_), len(matrix))
        self.assertEqual(len(np.unique(model.labels_)), 3)

    def test_dbscan_k_distances_grid_search_and_noise(self):
        """Kiểm tra đồ thị k-NN (loại tự khoảng cách 0), grid search và tách đúng điểm nhiễu nhãn -1."""
        cluster_a = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, -0.1]])
        cluster_b = np.array([[10.0, 10.0], [10.1, 10.1], [9.9, 10.1], [10.1, 9.9], [9.9, 9.9]])
        noise_point = np.array([[50.0, 50.0]])
        matrix = np.vstack([cluster_a, cluster_b, noise_point])

        trainer = DBSCANTrainer()
        k_dist_k1 = trainer.compute_k_distances(np.array([[0.0, 0.0], [3.0, 4.0], [20.0, 20.0]]), k=1)
        self.assertAlmostEqual(k_dist_k1[0], 5.0, places=6)
        self.assertAlmostEqual(k_dist_k1[1], 5.0, places=6)
        with self.assertRaises(ValueError):
            trainer.compute_k_distances(np.array([[0.0, 0.0], [1.0, 0.0]]), k=2)

        k_dist = trainer.compute_k_distances(matrix, k=3)
        self.assertEqual(len(k_dist), len(matrix))
        self.assertTrue(np.all(np.diff(k_dist) >= 0))
        self.assertGreater(k_dist[-1], 40.0)

        grid_df = trainer.grid_search_params(matrix, eps_values=[0.5, 1.0], min_samples_values=[3])
        self.assertEqual(len(grid_df), 2)
        self.assertTrue((grid_df["Số cụm"] == 2).all())
        self.assertTrue((grid_df["Số điểm nhiễu"] == 1).all())

        model = trainer.fit_model(matrix, eps=0.8, min_samples=3)
        self.assertEqual(model.labels_[-1], -1)
        self.assertEqual(len(set(model.labels_) - {-1}), 2)

    def test_kmedoids_centers_are_real_points_and_search_k(self):
        """Kiểm tra tâm cụm K-Medoids là điểm thật thuộc tập dữ liệu và tìm đúng K=2 cho 2 cụm."""
        matrix = np.array([
            [0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2],
            [10.0, 10.0], [10.2, 10.0], [10.0, 10.2], [10.2, 10.2],
        ])
        trainer = KMedoidsTrainer(n_clusters=2, random_state=42)
        labels = trainer.fit_predict(matrix)

        self.assertEqual(len(labels), 8)
        self.assertEqual(len(trainer.medoid_indices_), 2)
        for center in trainer.cluster_centers_:
            matches_real_point = any(np.allclose(center, row) for row in matrix)
            self.assertTrue(matches_real_point)
        self.assertEqual(len(set(labels[:4])), 1)
        self.assertEqual(len(set(labels[4:])), 1)
        self.assertNotEqual(labels[0], labels[4])

        k_search_df = trainer.search_optimal_k(matrix, k_range=range(2, 5))
        self.assertEqual(int(k_search_df.loc[k_search_df["Silhouette"].idxmax(), "K"]), 2)

    def test_persistence_save_and_load_roundtrip(self):
        """Kiểm tra lưu gói (model + scaler + metadata) xuống đĩa và nạp lại khớp dự đoán."""
        df = pd.DataFrame({
            "Age": [22.0, 24.0, 23.0, 55.0, 58.0, 56.0],
            "Income": [20000.0, 22000.0, 21000.0, 90000.0, 95000.0, 92000.0],
            "Total_Spending": [50.0, 60.0, 55.0, 1500.0, 1600.0, 1550.0],
        })
        scaler = FeatureScaler(method="standard")
        scaled_matrix = scaler.fit_transform(df)
        model = KMeansTrainer(random_state=42).fit_final_model(scaled_matrix, optimal_k=2)

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

            reloaded_labels = bundle["model"].predict(bundle["scaler"].transform(df))
            self.assertTrue(np.array_equal(model.labels_, reloaded_labels))

            with self.assertRaises(FileNotFoundError):
                load_model("khong_ton_tai.pkl", input_dir=out_dir)

    def test_duplicate_points_and_nan_guards_kmeans_kmedoids(self):
        """Kiểm tra từ chối K > số điểm duy nhất ở K-Means / K-Medoids và NaN đầu vào K-Medoids."""
        duplicate_matrix = np.array([[0.0, 0.0]] * 5 + [[10.0, 10.0]] * 5)
        with self.assertRaises(ValueError):
            KMeansTrainer(random_state=42).fit_final_model(duplicate_matrix, optimal_k=3)
        with self.assertRaises(ValueError):
            KMedoidsTrainer(n_clusters=3, random_state=42).fit_predict(duplicate_matrix)

        nan_matrix = np.array([[0.0, np.nan], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        with self.assertRaises(ValueError):
            KMedoidsTrainer(n_clusters=2, random_state=42).fit_predict(nan_matrix)

    def test_dbscan_search_clears_previous_best_params(self):
        matrix = np.array([[0., 0.], [0.1, 0.], [10., 10.], [10.1, 10.]])
        trainer = DBSCANTrainer()
        trainer.grid_search_params(matrix, [0.3], [2])
        self.assertEqual(trainer.best_params_, (0.3, 2))
        trainer.grid_search_params(matrix, [100.0], [2])
        self.assertIsNone(trainer.best_params_)
        self.assertEqual(trainer.grid_results_["eps"].tolist(), [100.0])


if __name__ == "__main__":
    unittest.main()

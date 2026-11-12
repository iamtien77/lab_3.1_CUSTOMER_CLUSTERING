"""
=============================================================================
KIỂM THỬ ĐƠN VỊ: MODULE ĐÁNH GIÁ PHÂN CỤM (TEST EVALUATION METRICS)
=============================================================================
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np

import pandas as pd

from src.evaluation.metrics import (
    compare_all_models,
    compute_dunn_index,
    evaluate_clustering,
)
from src.evaluation.profiling import (
    create_cluster_profiles,
    generate_customer_personas,
)
from src.evaluation.visualization import (
    plot_clusters_2d,
    plot_clusters_3d,
    plot_elbow_and_silhouette,
    plot_scaling_comparison,
    plot_silhouette_analysis,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Kiểm thử tại seam công khai cho các chỉ số đánh giá phân cụm."""

    def test_compute_dunn_index_worked_geometry(self):
        """Kiểm tra chỉ số Dunn khớp kết quả hình học tính tay (4.0 / 2.0 = 2.0)."""
        ma_tran = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.0, 2.0]])
        labels = np.array([0, 0, 1, 1])
        dunn = compute_dunn_index(ma_tran, labels)
        self.assertAlmostEqual(dunn, 2.0, places=6)

    def test_evaluate_clustering_valid_and_noise_handling(self):
        """Kiểm tra các chỉ số nội tại và loại trừ nhãn nhiễu -1 khi đánh giá."""
        ma_tran = np.array([
            [0.0, 0.0], [0.2, 0.1], [0.1, 0.2],
            [10.0, 10.0], [10.2, 10.1], [10.1, 10.2],
            [99.0, 99.0],
        ])
        labels = np.array([0, 0, 0, 1, 1, 1, -1])

        ket_qua = evaluate_clustering(ma_tran, labels, model_name="DBSCAN_Test")
        self.assertEqual(ket_qua["n_clusters"], 2)
        self.assertEqual(ket_qua["noise_count"], 1)
        self.assertAlmostEqual(ket_qua["noise_ratio_pct"], round(1 / 7 * 100, 2), places=2)
        self.assertTrue(0.8 <= ket_qua["silhouette"] <= 1.0)
        self.assertGreaterEqual(ket_qua["davies_bouldin"], 0.0)
        self.assertGreater(ket_qua["calinski_harabasz"], 0.0)
        self.assertEqual(ket_qua["negative_silhouette_pct"], 0.0)
        self.assertAlmostEqual(ket_qua["cluster_balance_ratio"], 1.0, places=6)

    def test_evaluate_clustering_invalid_partition_guards(self):
        """Đảm bảo không gán giá trị 0 giả tạo khi chỉ có 1 cụm, toàn nhiễu hoặc cụm đơn lẻ."""
        ma_tran = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        for nhan_loi in [
            np.array([0, 0, 0, 0]),      # Chỉ 1 cụm
            np.array([-1, -1, -1, -1]),  # Toàn bộ là nhiễu
            np.array([0, 1, 2, 3]),      # Số cụm == số mẫu
        ]:
            ket_qua = evaluate_clustering(ma_tran, nhan_loi, model_name="Invalid")
            self.assertTrue(np.isnan(ket_qua["silhouette"]))
            self.assertTrue(np.isnan(ket_qua["davies_bouldin"]))
            self.assertTrue(np.isnan(ket_qua["calinski_harabasz"]))

    def test_compare_all_models_exports_ranked_csv(self):
        """Kiểm tra bảng đối chiếu xếp hạng theo Silhouette giảm dần và xuất đúng file CSV."""
        ds = [
            evaluate_clustering(np.array([[0, 0], [0, 1], [10, 10], [10, 11]]), np.array([0, 0, 1, 1]), "Model_A"),
            evaluate_clustering(np.array([[0, 0], [5, 5], [1, 1], [6, 6]]), np.array([0, 0, 1, 1]), "Model_B"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "model_comparison.csv"
            bang = compare_all_models(ds, output_path=csv_path)
            self.assertTrue(csv_path.exists())
            self.assertEqual(len(bang), 2)
            self.assertEqual(bang.iloc[0]["Thuật toán"], "Model_A")

    def test_create_cluster_profiles_and_dynamic_personas(self):
        """Kiểm tra bảng hồ sơ cụm tính đúng trung bình/trung vị và sinh Persona động cho K bất kỳ."""
        df_kh = pd.DataFrame({
            "ID": [1, 2, 3, 4],
            "Age": [25.0, 35.0, 50.0, 60.0],
            "Income": [20000.0, 30000.0, 80000.0, 100000.0],
            "Total_Spending": [100.0, 200.0, 1500.0, 1900.0],
            "Education": ["Graduation", "Master", "PhD", "PhD"],
            "Marital_Status": ["Single", "Single", "Married", "Married"],
        })
        labels = [0, 0, 1, 1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = Path(tmp_dir) / "cluster_profiles.csv"
            profile_df = create_cluster_profiles(df_kh, labels, output_path=profile_path)
            self.assertTrue(profile_path.exists())
            self.assertEqual(len(profile_df), 2)
            self.assertEqual(int(profile_df["Số lượng"].sum()), 4)

            # Cụm 0: Age TB=30, Income TB=25000, Spending TB=150
            cum_0 = profile_df[profile_df["Cluster"] == 0].iloc[0]
            self.assertAlmostEqual(cum_0["Tuổi TB"], 30.0)
            self.assertAlmostEqual(cum_0["Thu nhập TB"], 25000.0)
            self.assertAlmostEqual(cum_0["Chi tiêu TB"], 150.0)
            self.assertAlmostEqual(cum_0["Tỷ lệ (%)"], 50.0)

            personas = generate_customer_personas(profile_df)
            self.assertEqual(len(personas), 2)
            ten_theo_cum = {p["cluster_id"]: p["persona_name"] for p in personas}
            self.assertIn("VIP", ten_theo_cum[1])
            self.assertIn("Tiết kiệm", ten_theo_cum[0])

    def test_visualization_exports_png_files(self):
        """Kiểm tra các hàm vẽ chính xuất đủ tệp PNG trong thư mục tạm không lỗi."""
        df_mau = pd.DataFrame({
            "ID": [1, 2, 3, 4],
            "Age": [24.0, 26.0, 52.0, 55.0],
            "Income": [22000.0, 24000.0, 88000.0, 92000.0],
            "Total_Spending": [80.0, 100.0, 1600.0, 1750.0],
        })
        labels = np.array([0, 0, 1, 1])
        centroids = np.array([[25.0, 23000.0, 90.0], [53.5, 90000.0, 1675.0]])
        k_df = pd.DataFrame({"K": [2, 3], "Inertia": [100.0, 40.0], "Silhouette": [0.85, 0.55]})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ma_tran_scaled = (df_mau[["Age", "Income", "Total_Spending"]] - df_mau[["Age", "Income", "Total_Spending"]].mean()) / df_mau[["Age", "Income", "Total_Spending"]].std(ddof=0)
            p1 = plot_elbow_and_silhouette(k_df, output_dir=tmp_dir)
            p2 = plot_clusters_2d(df_mau, labels, centroids=centroids, output_dir=tmp_dir)
            p3 = plot_clusters_3d(df_mau, labels, output_dir=tmp_dir)
            p4 = plot_silhouette_analysis(ma_tran_scaled.to_numpy(), labels, output_dir=tmp_dir)
            p5 = plot_scaling_comparison(df_mau, ma_tran_scaled.to_numpy(), output_dir=tmp_dir)
            for path_png in (p1, p2, p3, p4, p5):
                self.assertTrue(path_png.exists())
                self.assertGreater(path_png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

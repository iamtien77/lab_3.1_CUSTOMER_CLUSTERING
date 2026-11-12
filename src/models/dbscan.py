"""
=============================================================================
MODULE: PHÂN CỤM DỰA TRÊN MẬT ĐỘ DBSCAN (DENSITY-BASED SPATIAL CLUSTERING)
=============================================================================
Mục đích (Đáp ứng Bước 3 trong đề bài Practice Exercise 3):
- Phân cụm dựa trên mật độ điểm dữ liệu và tách các điểm nhiễu ngoại lai (nhãn -1).
- Tính khoảng cách k-NN giúp gợi ý bán kính lân cận eps qua đồ thị khuỷu tay.
- Thử nghiệm lưới tham số (eps, min_samples) để chọn cấu hình phù hợp nhất.
=============================================================================
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.evaluation.metrics import evaluate_clustering


class DBSCANTrainer:
    """
    Lớp quản lý phân cụm mật độ DBSCAN và dò tìm siêu tham số (eps, min_samples).
    """

    def __init__(self):
        """Khởi tạo các thuộc tính lưu trữ mô hình và kết quả tìm kiếm tham số."""
        self.model: Optional[DBSCAN] = None
        self.labels_: Optional[np.ndarray] = None
        self.grid_results_: Optional[pd.DataFrame] = None
        self.best_params_: Optional[Tuple[float, int]] = None

    def compute_k_distances(self, feature_matrix: np.ndarray, k: int = 4) -> np.ndarray:
        """
        Tính khoảng cách từ mỗi điểm đến láng giềng thứ k gần nhất (không tính chính nó) và sắp xếp tăng dần.
        """
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        if k < 1 or (k + 1) > len(matrix):
            raise ValueError(f"Giá trị k={k} yêu cầu ít nhất {k + 1} mẫu, hiện có {len(matrix)}.")

        neighbors_model = NearestNeighbors(n_neighbors=k + 1)
        neighbors_model.fit(matrix)
        distances, _ = neighbors_model.kneighbors(matrix)
        # Cột 0 là khoảng cách tới chính điểm đó (= 0), cột k là láng giềng thứ k thực sự
        k_distances = np.sort(distances[:, k])
        return k_distances

    def grid_search_params(
        self,
        feature_matrix: np.ndarray,
        eps_values: List[float],
        min_samples_values: List[int],
    ) -> pd.DataFrame:
        """
        Quét lưới cặp tham số (eps, min_samples), thống kê số cụm, tỷ lệ nhiễu và Silhouette (trên điểm không nhiễu).
        """
        self.best_params_ = None
        self.grid_results_ = None
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        grid_rows = []

        for eps in eps_values:
            if eps <= 0:
                continue
            for min_samples in min_samples_values:
                if min_samples < 1:
                    continue
                model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
                labels = model.fit_predict(matrix)
                metrics = evaluate_clustering(
                    matrix,
                    labels,
                    model_name=f"DBSCAN_eps{eps}_ms{min_samples}",
                    compute_dunn=False,
                )
                grid_rows.append(
                    {
                        "eps": float(eps),
                        "min_samples": int(min_samples),
                        "Số cụm": metrics["n_clusters"],
                        "Số điểm nhiễu": metrics["noise_count"],
                        "Tỷ lệ nhiễu %": metrics["noise_ratio_pct"],
                        "Silhouette": metrics["silhouette"],
                    }
                )

        if not grid_rows:
            raise ValueError("Không có cặp tham số (eps, min_samples) hợp lệ nào được truyền vào.")

        self.grid_results_ = pd.DataFrame(grid_rows)
        valid_rows = self.grid_results_.dropna(subset=["Silhouette"])
        if not valid_rows.empty:
            best_row = valid_rows.loc[valid_rows["Silhouette"].idxmax()]
            self.best_params_ = (float(best_row["eps"]), int(best_row["min_samples"]))
            print(
                f" [LƯỚI THAM SỐ DBSCAN] Gợi ý tốt nhất: eps={self.best_params_[0]}, "
                f"min_samples={self.best_params_[1]} | Cụm={int(best_row['Số cụm'])} | "
                f"Nhiễu={best_row['Tỷ lệ nhiễu %']:.2f}% | Silhouette={best_row['Silhouette']:.4f}"
            )
        else:
            print(" [CẢNH BÁO DBSCAN] Không có cấu hình nào trong lưới sinh từ 2 cụm hợp lệ trở lên.")
        return self.grid_results_

    def fit_model(self, feature_matrix: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
        """
        Huấn luyện mô hình DBSCAN với cặp tham số (eps, min_samples).
        """
        if eps <= 0 or min_samples < 1:
            raise ValueError(f"Tham số DBSCAN không hợp lệ: eps={eps}, min_samples={min_samples}.")

        matrix = np.asarray(feature_matrix, dtype=np.float64)
        self.model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        self.labels_ = self.model.fit_predict(matrix)
        n_clusters = len(set(self.labels_) - {-1})
        n_noise = int(np.sum(self.labels_ == -1))
        print(
            f" [HUẤN LUYỆN DBSCAN] Hoàn tất (eps={eps}, min_samples={min_samples}) -> "
            f"{n_clusters} cụm, {n_noise} điểm nhiễu ({n_noise / max(len(matrix), 1) * 100:.2f}%)."
        )
        return self.model

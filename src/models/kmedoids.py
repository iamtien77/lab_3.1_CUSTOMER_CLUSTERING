"""
=============================================================================
MODULE: PHÂN CỤM BẰNG K-MEDOIDS (BUILD TẤT ĐỊNH & CẬP NHẬT NỘI CỤM)
=============================================================================
Mục đích (Đáp ứng Bước 3 trong đề bài Practice Exercise 3):
- Cài đặt thuật toán K-Medoids (pha khởi tạo BUILD tất định kết hợp bước cập nhật
  tâm nội cụm kiểu Voronoi / Park & Jun) bằng NumPy/SciPy không cần thư viện ngoài.
- Tâm đại diện (Medoids) luôn là các điểm dữ liệu thật trong tập, giúp kháng
  ngoại lai tốt hơn so với tâm trung bình cộng của K-Means.
=============================================================================
"""

from typing import Iterable, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.config import K_SEARCH_RANGE, RANDOM_SEED
from src.evaluation.metrics import evaluate_clustering


_METRIC_MAP = {
    "euclidean": "euclidean",
    "manhattan": "cityblock",
}


class KMedoidsTrainer:
    """
    Lớp quản lý phân cụm K-Medoids chọn điểm dữ liệu thật làm tâm cụm.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = "euclidean",
        max_iter: int = 300,
        random_state: int = RANDOM_SEED,
    ):
        """Khởi tạo siêu tham số số cụm K, thước đo khoảng cách và giới hạn vòng lặp."""
        normalized_metric = metric.lower().strip()
        if normalized_metric not in _METRIC_MAP:
            raise ValueError(f"Thước đo '{metric}' không hợp lệ. Chỉ hỗ trợ: {list(_METRIC_MAP.keys())}.")

        self.n_clusters = int(n_clusters)
        self.metric = normalized_metric
        self._scipy_metric = _METRIC_MAP[normalized_metric]
        self.max_iter = int(max_iter)
        self.random_state = random_state

        self.medoid_indices_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.total_cost_: Optional[float] = None

    def _build_initial_medoids(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Pha BUILD của PAM: chọn medoid đầu có tổng khoảng cách nhỏ nhất, các medoid sau giảm chi phí tối đa.
        """
        n_samples = dist_matrix.shape[0]
        selected_indices: List[int] = [int(np.argmin(dist_matrix.sum(axis=1)))]
        nearest_distances = dist_matrix[:, selected_indices[0]].copy()

        while len(selected_indices) < self.n_clusters:
            # Ước lượng mức giảm tổng khoảng cách khi chọn thêm mỗi ứng viên
            cost_reductions = np.maximum(nearest_distances[:, None] - dist_matrix, 0.0).sum(axis=0)
            cost_reductions[selected_indices] = -1.0
            best_candidate = int(np.argmax(cost_reductions))
            if best_candidate in selected_indices:
                for idx in range(n_samples):
                    if idx not in selected_indices:
                        best_candidate = idx
                        break
            selected_indices.append(best_candidate)
            nearest_distances = np.minimum(nearest_distances, dist_matrix[:, best_candidate])

        return np.array(selected_indices, dtype=int)

    def fit_predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Huấn luyện K-Medoids (BUILD tất định & cập nhật medoid nội cụm) và trả về nhãn cụm.
        """
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"Ma trận đầu vào phải là 2 chiều, nhận được {matrix.ndim} chiều.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Ma trận đầu vào K-Medoids chứa giá trị NaN hoặc Inf.")

        n_samples = len(matrix)
        if n_samples < 3:
            raise ValueError(f"K-Medoids yêu cầu ít nhất 3 mẫu dữ liệu, nhận được {n_samples}.")

        unique_points_count = len(np.unique(matrix, axis=0))
        if self.n_clusters < 2 or self.n_clusters >= n_samples or self.n_clusters > unique_points_count:
            raise ValueError(
                f"Số cụm K={self.n_clusters} không hợp lệ với {n_samples} mẫu "
                f"({unique_points_count} điểm phân biệt)."
            )

        dist_matrix = cdist(matrix, matrix, metric=self._scipy_metric)
        medoids = self._build_initial_medoids(dist_matrix)
        labels = np.argmin(dist_matrix[:, medoids], axis=1)

        for _ in range(self.max_iter):
            new_medoids = medoids.copy()
            for cluster_id in range(self.n_clusters):
                member_indices = np.where(labels == cluster_id)[0]
                if len(member_indices) == 0:
                    continue
                sub_dist = dist_matrix[np.ix_(member_indices, member_indices)]
                intra_cluster_costs = sub_dist.sum(axis=1)
                new_medoids[cluster_id] = member_indices[int(np.argmin(intra_cluster_costs))]

            new_labels = np.argmin(dist_matrix[:, new_medoids], axis=1)
            if np.array_equal(new_medoids, medoids) and np.array_equal(new_labels, labels):
                break
            medoids = new_medoids
            labels = new_labels

        self.medoid_indices_ = medoids
        self.cluster_centers_ = matrix[medoids].copy()
        self.labels_ = labels
        self.total_cost_ = float(np.sum(np.min(dist_matrix[:, medoids], axis=1)))
        return self.labels_

    def predict(self, new_matrix: np.ndarray) -> np.ndarray:
        """Gán nhãn cụm cho dữ liệu mới dựa trên Medoid gần nhất."""
        if self.cluster_centers_ is None:
            raise RuntimeError("KMedoidsTrainer chưa được huấn luyện. Hãy gọi fit_predict() trước.")
        distances = cdist(np.asarray(new_matrix, dtype=np.float64), self.cluster_centers_, metric=self._scipy_metric)
        return np.argmin(distances, axis=1)

    def search_optimal_k(
        self,
        feature_matrix: np.ndarray,
        k_range: Iterable[int] = K_SEARCH_RANGE,
    ) -> pd.DataFrame:
        """
        Quét số cụm K với K-Medoids để ghi nhận tổng chi phí khoảng cách và Silhouette Score.
        """
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("Ma trận đầu vào search_optimal_k phải là mảng 2D hữu hạn.")
        unique_points_count = len(np.unique(matrix, axis=0))
        search_rows = []
        for k in k_range:
            if k < 2 or k >= len(matrix) or k > unique_points_count:
                continue
            trainer_k = KMedoidsTrainer(
                n_clusters=k,
                metric=self.metric,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            labels = trainer_k.fit_predict(matrix)
            metrics = evaluate_clustering(matrix, labels, model_name=f"KMedoids_K{k}", compute_dunn=False)
            search_rows.append(
                {
                    "K": int(k),
                    "Total_Cost": round(float(trainer_k.total_cost_), 4),
                    "Silhouette": metrics["silhouette"],
                    "Davies_Bouldin": metrics["davies_bouldin"],
                    "Calinski_Harabasz": metrics["calinski_harabasz"],
                }
            )

        search_df = pd.DataFrame(search_rows)
        if not search_df.empty:
            best_row = search_df.loc[search_df["Silhouette"].idxmax()]
            print(
                f" [QUÉT K-MEDOIDS] K tối ưu = {int(best_row['K'])} "
                f"(Silhouette = {best_row['Silhouette']:.4f}, Cost = {best_row['Total_Cost']:,.2f})"
            )
        return search_df

"""
=============================================================================
MODULE: PHÂN CỤM BẰNG K-MEANS (K-MEANS CLUSTERING)
=============================================================================
Mục đích (Đáp ứng Bước 3 trong đề bài Practice Exercise 3):
- Đóng vai trò Baseline Model chuẩn mực cho bài toán phân cụm khách hàng.
- Quét số cụm K từ 2 đến 10 với khởi tạo k-means++ để ghi nhận Inertia (WCSS)
  và các chỉ số phân tách (Silhouette, Davies-Bouldin, Calinski-Harabasz).
- Hỗ trợ thử nghiệm siêu tham số (init, n_init, max_iter) và huấn luyện mô hình cuối.
=============================================================================
"""

from typing import Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import K_SEARCH_RANGE, RANDOM_SEED
from src.evaluation.metrics import evaluate_clustering


class KMeansTrainer:
    """
    Lớp quản lý huấn luyện, dò tìm số cụm K và thử nghiệm tham số cho K-Means.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """Khởi tạo seed ngẫu nhiên để đảm bảo tính tái lập khi khởi tạo tâm cụm."""
        self.random_state = random_state
        self.model: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.search_results_: Optional[pd.DataFrame] = None

    @staticmethod
    def _validate_input_matrix(feature_matrix: np.ndarray) -> np.ndarray:
        """Kiểm tra ma trận đầu vào đủ số lượng mẫu (>= 3) và số điểm phân biệt (>= 2)."""
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"Ma trận đầu vào phải là 2 chiều, nhận được {matrix.ndim} chiều.")
        if len(matrix) < 3:
            raise ValueError(f"K-Means yêu cầu ít nhất 3 mẫu dữ liệu để phân cụm, nhận được {len(matrix)}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Ma trận đầu vào chứa giá trị NaN hoặc Inf.")
        if len(np.unique(matrix, axis=0)) < 2:
            raise ValueError("Dữ liệu chỉ có ít hơn 2 điểm phân biệt, không thể chia cụm.")
        return matrix

    def search_optimal_k(
        self,
        feature_matrix: np.ndarray,
        k_range: Iterable[int] = K_SEARCH_RANGE,
    ) -> pd.DataFrame:
        """
        Quét giá trị K trong k_range để tính WCSS (Inertia), Silhouette, Davies-Bouldin và Calinski-Harabasz.
        """
        matrix = self._validate_input_matrix(feature_matrix)
        unique_points_count = len(np.unique(matrix, axis=0))
        search_rows = []

        for k in k_range:
            if k < 2 or k >= len(matrix) or k > unique_points_count:
                print(f" [BỎ QUA K={k}] Không thỏa mãn 2 <= K < số mẫu ({len(matrix)}) và <= số điểm duy nhất ({unique_points_count}).")
                continue

            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=self.random_state,
            )
            labels = model.fit_predict(matrix)
            metrics = evaluate_clustering(matrix, labels, model_name=f"KMeans_K{k}", compute_dunn=False)

            search_rows.append(
                {
                    "K": int(k),
                    "Inertia": round(float(model.inertia_), 4),
                    "Silhouette": metrics["silhouette"],
                    "Davies_Bouldin": metrics["davies_bouldin"],
                    "Calinski_Harabasz": metrics["calinski_harabasz"],
                }
            )

        if not search_rows:
            raise ValueError("Không có giá trị K hợp lệ nào trong k_range để thử nghiệm.")

        self.search_results_ = pd.DataFrame(search_rows)
        best_row = self.search_results_.loc[self.search_results_["Silhouette"].idxmax()]
        print(
            f" [QUÉT K-MEANS] K tối ưu theo Silhouette Score = {int(best_row['K'])} "
            f"(Silhouette = {best_row['Silhouette']:.4f}, Inertia = {best_row['Inertia']:,.2f})"
        )
        return self.search_results_

    def experiment_hyperparameters(
        self,
        feature_matrix: np.ndarray,
        n_clusters: int,
        init_methods: tuple[str, ...] = ("k-means++", "random"),
        n_init_values: tuple[int, ...] = (10, 20),
        max_iter_values: tuple[int, ...] = (100, 300),
    ) -> pd.DataFrame:
        """
        Thử nghiệm các tổ hợp siêu tham số (init, n_init, max_iter) tại số cụm K đã chọn.
        """
        matrix = self._validate_input_matrix(feature_matrix)
        experiment_rows = []
        for init_method in init_methods:
            for n_init in n_init_values:
                for max_iter in max_iter_values:
                    model = KMeans(
                        n_clusters=n_clusters,
                        init=init_method,
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=self.random_state,
                    )
                    labels = model.fit_predict(matrix)
                    metrics = evaluate_clustering(
                        matrix,
                        labels,
                        model_name=f"KMeans_{init_method}_n{n_init}_it{max_iter}",
                        compute_dunn=False,
                    )
                    experiment_rows.append(
                        {
                            "init": init_method,
                            "n_init": n_init,
                            "max_iter": max_iter,
                            "Inertia": round(float(model.inertia_), 4),
                            "Silhouette": metrics["silhouette"],
                            "Davies_Bouldin": metrics["davies_bouldin"],
                            "n_iter_actual": int(model.n_iter_),
                        }
                    )
        return pd.DataFrame(experiment_rows)

    def fit_final_model(
        self,
        feature_matrix: np.ndarray,
        optimal_k: Optional[int] = None,
        init: str = "k-means++",
        n_init: int = 20,
        max_iter: int = 300,
    ) -> KMeans:
        """
        Huấn luyện mô hình K-Means cuối cùng. Nếu optimal_k=None, tự chọn K theo Silhouette cao nhất.
        """
        matrix = self._validate_input_matrix(feature_matrix)
        if optimal_k is None:
            k_search_df = self.search_optimal_k(matrix)
            optimal_k = int(k_search_df.loc[k_search_df["Silhouette"].idxmax(), "K"])

        unique_points_count = len(np.unique(matrix, axis=0))
        if optimal_k < 2 or optimal_k >= len(matrix) or optimal_k > unique_points_count:
            raise ValueError(
                f"Số cụm K={optimal_k} không hợp lệ với {len(matrix)} mẫu "
                f"({unique_points_count} điểm phân biệt)."
            )

        self.model = KMeans(
            n_clusters=optimal_k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state,
        )
        self.labels_ = self.model.fit_predict(matrix)
        self.cluster_centers_ = self.model.cluster_centers_
        print(
            f" [HUẤN LUYỆN K-MEANS] Hoàn tất với K={optimal_k}, init='{init}', "
            f"n_init={n_init}, hội tụ sau {self.model.n_iter_} vòng lặp."
        )
        return self.model

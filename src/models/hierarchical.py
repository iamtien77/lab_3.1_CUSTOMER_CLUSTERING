"""
=============================================================================
MODULE: PHÂN CỤM PHÂN CẤP (HIERARCHICAL / AGGLOMERATIVE CLUSTERING)
=============================================================================
Mục đích (Đáp ứng Bước 3 trong đề bài Practice Exercise 3):
- Xây dựng ma trận liên kết (Linkage Matrix) theo các phương pháp Ward, Complete, Average.
- Tính hệ số tương quan Cophenetic để đo mức độ bảo toàn cấu trúc khoảng cách trên cây.
- Huấn luyện mô hình AgglomerativeClustering để đối chiếu với K-Means Baseline.
=============================================================================
"""

from typing import Iterable, Optional
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

from src.config import K_SEARCH_RANGE
from src.evaluation.metrics import evaluate_clustering


_LINKAGE_METHODS = {"ward", "complete", "average", "single"}


class HierarchicalTrainer:
    """
    Lớp quản lý phân cụm phân cấp từ dưới lên (Agglomerative Hierarchical Clustering).
    """

    def __init__(self, linkage_method: str = "ward", metric: str = "euclidean"):
        """
        Khởi tạo phương pháp liên kết ('ward', 'complete', 'average', 'single') và khoảng cách.
        """
        normalized_method = linkage_method.lower().strip()
        if normalized_method not in _LINKAGE_METHODS:
            raise ValueError(f"Phương pháp linkage '{linkage_method}' không hợp lệ. Chọn từ {sorted(_LINKAGE_METHODS)}.")
        if normalized_method == "ward" and metric != "euclidean":
            raise ValueError("Phương pháp 'ward' bắt buộc sử dụng khoảng cách 'euclidean'.")

        self.linkage_method = normalized_method
        self.metric = metric
        self.linkage_matrix_: Optional[np.ndarray] = None
        self.cophenetic_corr_: Optional[float] = None
        self.model: Optional[AgglomerativeClustering] = None
        self.labels_: Optional[np.ndarray] = None

    def compute_linkage_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Tính ma trận liên kết phân cấp bằng SciPy và đo hệ số tương quan Cophenetic.
        """
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        if len(matrix) < 2:
            raise ValueError("Cần ít nhất 2 mẫu dữ liệu để dựng cây phân cấp.")

        linkage_matrix = linkage(matrix, method=self.linkage_method, metric=self.metric)
        cophenetic_corr, _ = cophenet(linkage_matrix, pdist(matrix, metric=self.metric))

        self.linkage_matrix_ = linkage_matrix
        self.cophenetic_corr_ = float(cophenetic_corr)
        print(
            f" [CÂY PHÂN CẤP - {self.linkage_method.upper()}] "
            f"Hệ số tương quan Cophenetic = {self.cophenetic_corr_:.4f}"
        )
        return linkage_matrix

    def compute_cut_threshold(self, n_clusters: int) -> float:
        """
        Tính ngưỡng cắt trên ma trận liên kết sao cho thu được đúng n_clusters cụm.
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Cần gọi compute_linkage_matrix() trước khi tính ngưỡng cắt.")
        n_merges = len(self.linkage_matrix_)
        if n_clusters < 2 or n_clusters > n_merges:
            raise ValueError(f"Số cụm n_clusters={n_clusters} không hợp lệ cho ma trận liên kết.")
        return float((self.linkage_matrix_[-n_clusters, 2] + self.linkage_matrix_[-(n_clusters - 1), 2]) / 2.0)

    def search_optimal_k(
        self,
        feature_matrix: np.ndarray,
        k_range: Iterable[int] = K_SEARCH_RANGE,
    ) -> pd.DataFrame:
        """
        Quét số cụm K với AgglomerativeClustering để đối chiếu chất lượng phân tách (không dùng Inertia).
        """
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        search_rows = []
        for k in k_range:
            if k < 2 or k >= len(matrix):
                continue
            model = self.fit_model(matrix, n_clusters=k)
            metrics = evaluate_clustering(
                matrix,
                model.labels_,
                model_name=f"Hierarchical_{self.linkage_method}_K{k}",
                compute_dunn=False,
            )
            search_rows.append(
                {
                    "K": int(k),
                    "Silhouette": metrics["silhouette"],
                    "Davies_Bouldin": metrics["davies_bouldin"],
                    "Calinski_Harabasz": metrics["calinski_harabasz"],
                }
            )
        return pd.DataFrame(search_rows)

    def fit_model(self, feature_matrix: np.ndarray, n_clusters: int = 3) -> AgglomerativeClustering:
        """
        Huấn luyện mô hình AgglomerativeClustering với số cụm n_clusters chỉ định.
        """
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        if n_clusters < 2 or n_clusters >= len(matrix):
            raise ValueError(f"Số cụm K={n_clusters} không hợp lệ với tập có {len(matrix)} mẫu.")

        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage_method,
            metric=self.metric,
        )
        self.labels_ = self.model.fit_predict(matrix)
        print(
            f" [PHÂN CỤM PHÂN CẤP] Hoàn tất Agglomerative (linkage='{self.linkage_method}', K={n_clusters})."
        )
        return self.model

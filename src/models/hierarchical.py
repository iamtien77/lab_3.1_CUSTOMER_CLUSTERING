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
        phuong_phap = linkage_method.lower().strip()
        if phuong_phap not in _LINKAGE_METHODS:
            raise ValueError(f"Phương pháp linkage '{linkage_method}' không hợp lệ. Chọn từ {sorted(_LINKAGE_METHODS)}.")
        if phuong_phap == "ward" and metric != "euclidean":
            raise ValueError("Phương pháp 'ward' bắt buộc sử dụng khoảng cách 'euclidean'.")

        self.linkage_method = phuong_phap
        self.metric = metric
        self.linkage_matrix_: Optional[np.ndarray] = None
        self.cophenetic_corr_: Optional[float] = None
        self.model: Optional[AgglomerativeClustering] = None
        self.labels_: Optional[np.ndarray] = None

    def compute_linkage_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Tính ma trận liên kết phân cấp bằng SciPy và đo hệ số tương quan Cophenetic.
        """
        ma_tran = np.asarray(X, dtype=np.float64)
        if len(ma_tran) < 2:
            raise ValueError("Cần ít nhất 2 mẫu dữ liệu để dựng cây phân cấp.")

        ma_tran_lien_ket = linkage(ma_tran, method=self.linkage_method, metric=self.metric)
        tuong_quan_coph, _ = cophenet(ma_tran_lien_ket, pdist(ma_tran, metric=self.metric))

        self.linkage_matrix_ = ma_tran_lien_ket
        self.cophenetic_corr_ = float(tuong_quan_coph)
        print(
            f" [CÂY PHÂN CẤP - {self.linkage_method.upper()}] "
            f"Hệ số tương quan Cophenetic = {self.cophenetic_corr_:.4f}"
        )
        return ma_tran_lien_ket

    def compute_cut_threshold(self, n_clusters: int) -> float:
        """
        Tính ngưỡng cắt trên ma trận liên kết sao cho thu được đúng n_clusters cụm.
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Cần gọi compute_linkage_matrix() trước khi tính ngưỡng cắt.")
        so_buoc = len(self.linkage_matrix_)
        if n_clusters < 2 or n_clusters > so_buoc:
            raise ValueError(f"Số cụm n_clusters={n_clusters} không hợp lệ cho ma trận liên kết.")
        return float((self.linkage_matrix_[-n_clusters, 2] + self.linkage_matrix_[-(n_clusters - 1), 2]) / 2.0)

    def search_optimal_k(
        self,
        X: np.ndarray,
        k_range: Iterable[int] = K_SEARCH_RANGE,
    ) -> pd.DataFrame:
        """
        Quét số cụm K với AgglomerativeClustering để đối chiếu chất lượng phân tách (không dùng Inertia).
        """
        ma_tran = np.asarray(X, dtype=np.float64)
        danh_sach = []
        for k in k_range:
            if k < 2 or k >= len(ma_tran):
                continue
            model = self.fit_model(ma_tran, n_clusters=k)
            chi_so = evaluate_clustering(
                ma_tran,
                model.labels_,
                model_name=f"Hierarchical_{self.linkage_method}_K{k}",
                compute_dunn=False,
            )
            danh_sach.append(
                {
                    "K": int(k),
                    "Silhouette": chi_so["silhouette"],
                    "Davies_Bouldin": chi_so["davies_bouldin"],
                    "Calinski_Harabasz": chi_so["calinski_harabasz"],
                }
            )
        return pd.DataFrame(danh_sach)

    def fit_model(self, X: np.ndarray, n_clusters: int = 3) -> AgglomerativeClustering:
        """
        Huấn luyện mô hình AgglomerativeClustering với số cụm n_clusters chỉ định.
        """
        ma_tran = np.asarray(X, dtype=np.float64)
        if n_clusters < 2 or n_clusters >= len(ma_tran):
            raise ValueError(f"Số cụm K={n_clusters} không hợp lệ với tập có {len(ma_tran)} mẫu.")

        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage_method,
            metric=self.metric,
        )
        self.labels_ = self.model.fit_predict(ma_tran)
        print(
            f" [PHÂN CỤM PHÂN CẤP] Hoàn tất Agglomerative (linkage='{self.linkage_method}', K={n_clusters})."
        )
        return self.model

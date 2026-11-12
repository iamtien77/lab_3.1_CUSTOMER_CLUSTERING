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
        thuoc_do = metric.lower().strip()
        if thuoc_do not in _METRIC_MAP:
            raise ValueError(f"Thước đo '{metric}' không hợp lệ. Chỉ hỗ trợ: {list(_METRIC_MAP.keys())}.")

        self.n_clusters = int(n_clusters)
        self.metric = thuoc_do
        self._scipy_metric = _METRIC_MAP[thuoc_do]
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
        so_mau = dist_matrix.shape[0]
        chi_so_chon: List[int] = [int(np.argmin(dist_matrix.sum(axis=1)))]
        kc_gan_nhat = dist_matrix[:, chi_so_chon[0]].copy()

        while len(chi_so_chon) < self.n_clusters:
            # Ước lượng mức giảm tổng khoảng cách khi chọn thêm mỗi ứng viên
            loi_ich = np.maximum(kc_gan_nhat[:, None] - dist_matrix, 0.0).sum(axis=0)
            loi_ich[chi_so_chon] = -1.0
            ung_vien_tot_nhat = int(np.argmax(loi_ich))
            if ung_vien_tot_nhat in chi_so_chon:
                for idx in range(so_mau):
                    if idx not in chi_so_chon:
                        ung_vien_tot_nhat = idx
                        break
            chi_so_chon.append(ung_vien_tot_nhat)
            kc_gan_nhat = np.minimum(kc_gan_nhat, dist_matrix[:, ung_vien_tot_nhat])

        return np.array(chi_so_chon, dtype=int)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Huấn luyện K-Medoids (BUILD tất định & cập nhật medoid nội cụm) và trả về nhãn cụm.
        """
        ma_tran = np.asarray(X, dtype=np.float64)
        if ma_tran.ndim != 2:
            raise ValueError(f"Ma trận đầu vào phải là 2 chiều, nhận được {ma_tran.ndim} chiều.")
        if not np.all(np.isfinite(ma_tran)):
            raise ValueError("Ma trận đầu vào K-Medoids chứa giá trị NaN hoặc Inf.")

        so_mau = len(ma_tran)
        if so_mau < 3:
            raise ValueError(f"K-Medoids yêu cầu ít nhất 3 mẫu dữ liệu, nhận được {so_mau}.")

        so_diem_duy_nhat = len(np.unique(ma_tran, axis=0))
        if self.n_clusters < 2 or self.n_clusters >= so_mau or self.n_clusters > so_diem_duy_nhat:
            raise ValueError(
                f"Số cụm K={self.n_clusters} không hợp lệ với {so_mau} mẫu "
                f"({so_diem_duy_nhat} điểm phân biệt)."
            )

        dist_matrix = cdist(ma_tran, ma_tran, metric=self._scipy_metric)
        medoids = self._build_initial_medoids(dist_matrix)
        labels = np.argmin(dist_matrix[:, medoids], axis=1)

        for _ in range(self.max_iter):
            medoids_moi = medoids.copy()
            for cum_id in range(self.n_clusters):
                thanh_vien = np.where(labels == cum_id)[0]
                if len(thanh_vien) == 0:
                    continue
                sub_dist = dist_matrix[np.ix_(thanh_vien, thanh_vien)]
                chi_phi_noi_cum = sub_dist.sum(axis=1)
                medoids_moi[cum_id] = thanh_vien[int(np.argmin(chi_phi_noi_cum))]

            nhan_moi = np.argmin(dist_matrix[:, medoids_moi], axis=1)
            if np.array_equal(medoids_moi, medoids) and np.array_equal(nhan_moi, labels):
                break
            medoids = medoids_moi
            labels = nhan_moi

        self.medoid_indices_ = medoids
        self.cluster_centers_ = ma_tran[medoids].copy()
        self.labels_ = labels
        self.total_cost_ = float(np.sum(np.min(dist_matrix[:, medoids], axis=1)))
        return self.labels_

    def predict(self, ma_tran_moi: np.ndarray) -> np.ndarray:
        """Gán nhãn cụm cho dữ liệu mới dựa trên Medoid gần nhất."""
        if self.cluster_centers_ is None:
            raise RuntimeError("KMedoidsTrainer chưa được huấn luyện. Hãy gọi fit_predict() trước.")
        khoang_cach = cdist(np.asarray(ma_tran_moi, dtype=np.float64), self.cluster_centers_, metric=self._scipy_metric)
        return np.argmin(khoang_cach, axis=1)

    def search_optimal_k(
        self,
        X: np.ndarray,
        k_range: Iterable[int] = K_SEARCH_RANGE,
    ) -> pd.DataFrame:
        """
        Quét số cụm K với K-Medoids để ghi nhận tổng chi phí khoảng cách và Silhouette Score.
        """
        ma_tran = np.asarray(X, dtype=np.float64)
        if ma_tran.ndim != 2 or not np.all(np.isfinite(ma_tran)):
            raise ValueError("Ma trận đầu vào search_optimal_k phải là mảng 2D hữu hạn.")
        so_diem_duy_nhat = len(np.unique(ma_tran, axis=0))
        ket_qua = []
        for k in k_range:
            if k < 2 or k >= len(ma_tran) or k > so_diem_duy_nhat:
                continue
            trainer_k = KMedoidsTrainer(
                n_clusters=k,
                metric=self.metric,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            nhan = trainer_k.fit_predict(ma_tran)
            chi_so = evaluate_clustering(ma_tran, nhan, model_name=f"KMedoids_K{k}", compute_dunn=False)
            ket_qua.append(
                {
                    "K": int(k),
                    "Total_Cost": round(float(trainer_k.total_cost_), 4),
                    "Silhouette": chi_so["silhouette"],
                    "Davies_Bouldin": chi_so["davies_bouldin"],
                    "Calinski_Harabasz": chi_so["calinski_harabasz"],
                }
            )

        bang = pd.DataFrame(ket_qua)
        if not bang.empty:
            tot_nhat = bang.loc[bang["Silhouette"].idxmax()]
            print(
                f" [QUÉT K-MEDOIDS] K tối ưu = {int(tot_nhat['K'])} "
                f"(Silhouette = {tot_nhat['Silhouette']:.4f}, Cost = {tot_nhat['Total_Cost']:,.2f})"
            )
        return bang

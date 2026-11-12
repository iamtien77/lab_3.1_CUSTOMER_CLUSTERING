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
    def _validate_input_matrix(X: np.ndarray) -> np.ndarray:
        """Kiểm tra ma trận đầu vào đủ số lượng mẫu (>= 3) và số điểm phân biệt (>= 2)."""
        ma_tran = np.asarray(X, dtype=np.float64)
        if ma_tran.ndim != 2:
            raise ValueError(f"Ma trận đầu vào phải là 2 chiều, nhận được {ma_tran.ndim} chiều.")
        if len(ma_tran) < 3:
            raise ValueError(f"K-Means yêu cầu ít nhất 3 mẫu dữ liệu để phân cụm, nhận được {len(ma_tran)}.")
        if not np.all(np.isfinite(ma_tran)):
            raise ValueError("Ma trận đầu vào chứa giá trị NaN hoặc Inf.")
        if len(np.unique(ma_tran, axis=0)) < 2:
            raise ValueError("Dữ liệu chỉ có ít hơn 2 điểm phân biệt, không thể chia cụm.")
        return ma_tran

    def search_optimal_k(
        self,
        X: np.ndarray,
        k_range: Iterable[int] = K_SEARCH_RANGE,
    ) -> pd.DataFrame:
        """
        Quét giá trị K trong k_range để tính WCSS (Inertia), Silhouette, Davies-Bouldin và Calinski-Harabasz.
        """
        ma_tran = self._validate_input_matrix(X)
        so_diem_duy_nhat = len(np.unique(ma_tran, axis=0))
        ket_qua_dong = []

        for k in k_range:
            if k < 2 or k >= len(ma_tran) or k > so_diem_duy_nhat:
                print(f" [BỎ QUA K={k}] Không thỏa mãn 2 <= K < số mẫu ({len(ma_tran)}) và <= số điểm duy nhất ({so_diem_duy_nhat}).")
                continue

            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=self.random_state,
            )
            nhan = model.fit_predict(ma_tran)
            chi_so = evaluate_clustering(ma_tran, nhan, model_name=f"KMeans_K{k}", compute_dunn=False)

            ket_qua_dong.append(
                {
                    "K": int(k),
                    "Inertia": round(float(model.inertia_), 4),
                    "Silhouette": chi_so["silhouette"],
                    "Davies_Bouldin": chi_so["davies_bouldin"],
                    "Calinski_Harabasz": chi_so["calinski_harabasz"],
                }
            )

        if not ket_qua_dong:
            raise ValueError("Không có giá trị K hợp lệ nào trong k_range để thử nghiệm.")

        self.search_results_ = pd.DataFrame(ket_qua_dong)
        dong_tot_nhat = self.search_results_.loc[self.search_results_["Silhouette"].idxmax()]
        print(
            f" [QUÉT K-MEANS] K tối ưu theo Silhouette Score = {int(dong_tot_nhat['K'])} "
            f"(Silhouette = {dong_tot_nhat['Silhouette']:.4f}, Inertia = {dong_tot_nhat['Inertia']:,.2f})"
        )
        return self.search_results_

    def experiment_hyperparameters(
        self,
        X: np.ndarray,
        n_clusters: int,
        init_methods: tuple[str, ...] = ("k-means++", "random"),
        n_init_values: tuple[int, ...] = (10, 20),
        max_iter_values: tuple[int, ...] = (100, 300),
    ) -> pd.DataFrame:
        """
        Thử nghiệm các tổ hợp siêu tham số (init, n_init, max_iter) tại số cụm K đã chọn.
        """
        ma_tran = self._validate_input_matrix(X)
        danh_sach_thu = []
        for init_pp in init_methods:
            for n_init in n_init_values:
                for max_iter in max_iter_values:
                    model = KMeans(
                        n_clusters=n_clusters,
                        init=init_pp,
                        n_init=n_init,
                        max_iter=max_iter,
                        random_state=self.random_state,
                    )
                    nhan = model.fit_predict(ma_tran)
                    chi_so = evaluate_clustering(
                        ma_tran,
                        nhan,
                        model_name=f"KMeans_{init_pp}_n{n_init}_it{max_iter}",
                        compute_dunn=False,
                    )
                    danh_sach_thu.append(
                        {
                            "init": init_pp,
                            "n_init": n_init,
                            "max_iter": max_iter,
                            "Inertia": round(float(model.inertia_), 4),
                            "Silhouette": chi_so["silhouette"],
                            "Davies_Bouldin": chi_so["davies_bouldin"],
                            "n_iter_actual": int(model.n_iter_),
                        }
                    )
        return pd.DataFrame(danh_sach_thu)

    def fit_final_model(
        self,
        X: np.ndarray,
        optimal_k: Optional[int] = None,
        init: str = "k-means++",
        n_init: int = 20,
        max_iter: int = 300,
    ) -> KMeans:
        """
        Huấn luyện mô hình K-Means cuối cùng. Nếu optimal_k=None, tự chọn K theo Silhouette cao nhất.
        """
        ma_tran = self._validate_input_matrix(X)
        if optimal_k is None:
            bang_k = self.search_optimal_k(ma_tran)
            optimal_k = int(bang_k.loc[bang_k["Silhouette"].idxmax(), "K"])

        so_diem_duy_nhat = len(np.unique(ma_tran, axis=0))
        if optimal_k < 2 or optimal_k >= len(ma_tran) or optimal_k > so_diem_duy_nhat:
            raise ValueError(
                f"Số cụm K={optimal_k} không hợp lệ với {len(ma_tran)} mẫu "
                f"({so_diem_duy_nhat} điểm phân biệt)."
            )

        self.model = KMeans(
            n_clusters=optimal_k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state,
        )
        self.labels_ = self.model.fit_predict(ma_tran)
        self.cluster_centers_ = self.model.cluster_centers_
        print(
            f" [HUẤN LUYỆN K-MEANS] Hoàn tất với K={optimal_k}, init='{init}', "
            f"n_init={n_init}, hội tụ sau {self.model.n_iter_} vòng lặp."
        )
        return self.model

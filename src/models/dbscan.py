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

    def compute_k_distances(self, X: np.ndarray, k: int = 4) -> np.ndarray:
        """
        Tính khoảng cách từ mỗi điểm đến láng giềng thứ k gần nhất (không tính chính nó) và sắp xếp tăng dần.
        """
        ma_tran = np.asarray(X, dtype=np.float64)
        if k < 1 or (k + 1) > len(ma_tran):
            raise ValueError(f"Giá trị k={k} yêu cầu ít nhất {k + 1} mẫu, hiện có {len(ma_tran)}.")

        tim_lang_gieng = NearestNeighbors(n_neighbors=k + 1)
        tim_lang_gieng.fit(ma_tran)
        khoang_cach, _ = tim_lang_gieng.kneighbors(ma_tran)
        # Cột 0 là khoảng cách tới chính điểm đó (= 0), cột k là láng giềng thứ k thực sự
        khoang_cach_thu_k = np.sort(khoang_cach[:, k])
        return khoang_cach_thu_k

    def grid_search_params(
        self,
        X: np.ndarray,
        eps_values: List[float],
        min_samples_values: List[int],
    ) -> pd.DataFrame:
        """
        Quét lưới cặp tham số (eps, min_samples), thống kê số cụm, tỷ lệ nhiễu và Silhouette (trên điểm không nhiễu).
        """
        ma_tran = np.asarray(X, dtype=np.float64)
        ket_qua_luoi = []

        for eps in eps_values:
            if eps <= 0:
                continue
            for min_samples in min_samples_values:
                if min_samples < 1:
                    continue
                model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
                nhan = model.fit_predict(ma_tran)
                chi_so = evaluate_clustering(
                    ma_tran,
                    nhan,
                    model_name=f"DBSCAN_eps{eps}_ms{min_samples}",
                    compute_dunn=False,
                )
                ket_qua_luoi.append(
                    {
                        "eps": float(eps),
                        "min_samples": int(min_samples),
                        "Số cụm": chi_so["n_clusters"],
                        "Số điểm nhiễu": chi_so["noise_count"],
                        "Tỷ lệ nhiễu %": chi_so["noise_ratio_pct"],
                        "Silhouette": chi_so["silhouette"],
                    }
                )

        if not ket_qua_luoi:
            raise ValueError("Không có cặp tham số (eps, min_samples) hợp lệ nào được truyền vào.")

        self.grid_results_ = pd.DataFrame(ket_qua_luoi)
        hop_le = self.grid_results_.dropna(subset=["Silhouette"])
        if not hop_le.empty:
            tot_nhat = hop_le.loc[hop_le["Silhouette"].idxmax()]
            self.best_params_ = (float(tot_nhat["eps"]), int(tot_nhat["min_samples"]))
            print(
                f" [LƯỚI THAM SỐ DBSCAN] Gợi ý tốt nhất: eps={self.best_params_[0]}, "
                f"min_samples={self.best_params_[1]} | Cụm={int(tot_nhat['Số cụm'])} | "
                f"Nhiễu={tot_nhat['Tỷ lệ nhiễu %']:.2f}% | Silhouette={tot_nhat['Silhouette']:.4f}"
            )
        else:
            print(" [CẢNH BÁO DBSCAN] Không có cấu hình nào trong lưới sinh từ 2 cụm hợp lệ trở lên.")
        return self.grid_results_

    def fit_model(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
        """
        Huấn luyện mô hình DBSCAN với cặp tham số (eps, min_samples).
        """
        if eps <= 0 or min_samples < 1:
            raise ValueError(f"Tham số DBSCAN không hợp lệ: eps={eps}, min_samples={min_samples}.")

        ma_tran = np.asarray(X, dtype=np.float64)
        self.model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        self.labels_ = self.model.fit_predict(ma_tran)
        so_cum = len(set(self.labels_) - {-1})
        so_nhieu = int(np.sum(self.labels_ == -1))
        print(
            f" [HUẤN LUYỆN DBSCAN] Hoàn tất (eps={eps}, min_samples={min_samples}) -> "
            f"{so_cum} cụm, {so_nhieu} điểm nhiễu ({so_nhieu / max(len(ma_tran), 1) * 100:.2f}%)."
        )
        return self.model

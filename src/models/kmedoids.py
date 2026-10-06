"""
=============================================================================
MODULE: PHÂN CỤM BẰNG K-MEDOIDS (PAM - PARTITIONING AROUND MEDOIDS)
=============================================================================
Mục đích (Đáp ứng Bước 3 trong đề bài Practice Exercise 3):
- Phân cụm bằng thuật toán K-Medoids (Partitioning Around Medoids).
- Khác với K-Means (tâm cụm là điểm trung bình ảo giả định), K-Medoids chọn chính các điểm
  dữ liệu có thật trong tập làm tâm đại diện (Medoids).
- Nhờ đó, K-Medoids cực kỳ bền vững và ít bị ảnh hưởng bởi các giá trị ngoại lai (Outliers)
  hoặc phân phối dữ liệu bị lệch.
=============================================================================
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.config import RANDOM_SEED


class KMedoidsTrainer:
    """
    Lớp quản lý phân cụm K-Medoids (PAM algorithm).
    """

    def __init__(
        self,
        n_clusters: int = 5,
        metric: str = "euclidean",
        max_iter: int = 300,
        random_state: int = RANDOM_SEED,
    ):
        """
        Ý LÀM:
        1. Nhận vào số cụm n_clusters (mặc định 5).
        2. Nhận vào khoảng cách metric: 'euclidean' hoặc 'manhattan'.
        3. Khởi tạo các thuộc tính lưu trữ medoid_indices_, cluster_centers_, labels_.
        """
        pass

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ý LÀM (Thuật toán PAM Build & Swap):
        1. Bước Khởi tạo (BUILD phase):
           - Chọn K điểm dữ liệu đầu tiên làm K Medoids ban đầu (có thể chọn ngẫu nhiên
             hoặc chọn điểm có tổng khoảng cách tới các điểm khác nhỏ nhất).
        2. Bước Lặp Tối Ưu (SWAP phase):
           - Gán mỗi điểm dữ liệu vào Medoid gần nhất để tạo thành K cụm.
           - Tính tổng chi phí (tổng khoảng cách từ các điểm tới Medoid của cụm đó).
           - Với mỗi Medoid, thử tráo đổi (swap) với một điểm không phải Medoid trong cụm.
           - Nếu việc tráo đổi làm giảm tổng chi phí: chấp nhận Medoid mới.
           - Lặp lại cho đến khi chi phí không giảm thêm hoặc đạt max_iter.
        3. Lưu lại chỉ số của các Medoid (medoid_indices_) và nhãn phân cụm (labels_).
        4. Trả về mảng nhãn cụm của từng mẫu.
        """
        pass

    def search_optimal_k(self, X: np.ndarray, k_range: range = range(2, 11)) -> pd.DataFrame:
        """
        Ý LÀM:
        1. Chạy K-Medoids với K từ 2 đến 10.
        2. Đo lường Silhouette Score và tổng chi phí khoảng cách cho từng K.
        3. Trả về DataFrame so sánh để chọn K tối ưu cho K-Medoids.
        """
        pass

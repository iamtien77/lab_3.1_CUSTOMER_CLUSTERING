"""
=============================================================================
MODULE: PHÂN CỤM BẰNG K-MEANS (K-MEANS CLUSTERING)
=============================================================================
Mục đích (Đáp ứng Bước 3 trong đề bài Practice Exercise 3):
- Thực hiện phân cụm K-Means theo phương pháp khởi tạo chuẩn k-means++.
- Quét số cụm K từ 2 đến 10 để tính toán:
  * Điểm quán tính WCSS (Inertia) nhằm vẽ biểu đồ khuỷu tay (Elbow Method).
  * Hệ số Silhouette để tìm số cụm K tối ưu nhất.
- Thử nghiệm các siêu tham số (init='k-means++', 'random', n_init, max_iter).
- Huấn luyện mô hình K-Means cuối cùng và trích xuất tâm cụm (Centroids).
=============================================================================
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.config import RANDOM_SEED, K_SEARCH_RANGE


class KMeansTrainer:
    """
    Lớp quản lý huấn luyện, quét tham số K và đánh giá mô hình K-Means.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Ý LÀM:
        Khởi tạo seed ngẫu nhiên để đảm bảo tính tái lập (reproducibility).
        """
        pass

    def search_optimal_k(
        self,
        X: np.ndarray,
        k_range: range = K_SEARCH_RANGE,
    ) -> pd.DataFrame:
        """
        Ý LÀM (Elbow Method & Silhouette Search):
        1. Duyệt qua từng giá trị K trong k_range (ví dụ từ 2 đến 10).
        2. Với mỗi K:
           - Khởi tạo KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=self.random_state).
           - Gọi fit(X).
           - Lấy giá trị WCSS / Inertia (độ nén nội cụm: tổng bình phương khoảng cách tới tâm).
           - Tính hệ số Silhouette Score (đo độ gắn kết nội cụm so với độ tách rời giữa các cụm).
           - Tính Davies-Bouldin Index và Calinski-Harabasz Score.
        3. Tổng hợp kết quả thành DataFrame bảng đối chiếu:
           [K, Inertia, Silhouette, Davies_Bouldin, Calinski_Harabasz].
        4. Trả về DataFrame kết quả để vẽ đồ thị Elbow và xác định K tối ưu.
        """
        pass

    def fit_final_model(
        self,
        X: np.ndarray,
        optimal_k: int = 5,
        init: str = "k-means++",
        n_init: int = 20,
    ) -> KMeans:
        """
        Ý LÀM:
        1. Khởi tạo mô hình KMeans hoàn chỉnh với số cụm optimal_k đã chọn.
        2. Cấu hình tham số tối ưu: init='k-means++', n_init=20, max_iter=300.
        3. Fit mô hình trên dữ liệu X.
        4. Lưu trữ nhãn cụm (labels_) và tọa độ tâm cụm (cluster_centers_).
        5. Trả về đối tượng mô hình KMeans đã huấn luyện.
        """
        pass

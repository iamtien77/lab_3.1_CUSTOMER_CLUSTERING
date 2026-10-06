"""
=============================================================================
MODULE: PHÂN CỤM DỰA TRÊN MẬT ĐỘ DBSCAN (DENSITY-BASED SPATIAL CLUSTERING)
=============================================================================
Mục đích (Đáp ứng Bước 3 trong đề bài Practice Exercise 3):
- Phân cụm dựa trên mật độ điểm dữ liệu, có khả năng phát hiện các cụm mang hình dạng tùy ý.
- Tìm giá trị bán kính lân cận (eps) tối ưu bằng thuật toán k-nearest neighbors distance graph.
- Tự động phát hiện và tách biệt các điểm nhiễu ngoại lai (Noise points có nhãn -1).
- Thử nghiệm lưới tham số (Grid Search) trên cặp tham số (eps, min_samples).
=============================================================================
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class DBSCANTrainer:
    """
    Lớp quản lý phân cụm mật độ DBSCAN và dò tìm siêu tham số (eps, min_samples).
    """

    def __init__(self):
        """
        Ý LÀM:
        Khởi tạo các thuộc tính lưu trữ mô hình và kết quả phân tích.
        """
        pass

    def compute_k_distances(self, X: np.ndarray, k: int = 4) -> np.ndarray:
        """
        Ý LÀM (K-NN Distance Graph tìm điểm khuỷu tay cho eps):
        1. Sử dụng NearestNeighbors(n_neighbors=k).
        2. Tính khoảng cách từ mỗi điểm tới láng giềng thứ k gần nó nhất.
        3. Sắp xếp toàn bộ các khoảng cách này theo thứ tự tăng dần.
        4. Trả về mảng khoảng cách đã sắp xếp để vẽ biểu đồ tìm điểm 'khuỷu tay' (Elbow)
           làm gợi ý chọn bán kính eps thích hợp.
        """
        pass

    def grid_search_params(
        self,
        X: np.ndarray,
        eps_values: List[float],
        min_samples_values: List[int],
    ) -> pd.DataFrame:
        """
        Ý LÀM (Quét lưới tham số eps và min_samples):
        1. Duyệt qua từng cặp giá trị (eps, min_samples).
        2. Với mỗi cặp:
           - Khởi tạo DBSCAN(eps=eps, min_samples=min_samples).
           - Gọi fit(X) và lấy nhãn cụm (labels_).
           - Đếm số lượng cụm sinh ra (loại bỏ nhãn nhiễu -1).
           - Đếm số lượng và tỷ lệ điểm nhiễu (noise ratio %).
           - Nếu số lượng cụm >= 2: tính điểm Silhouette Score trên các điểm không phải nhiễu.
        3. Lưu kết quả vào bảng tổng hợp: [eps, min_samples, Số cụm, Số điểm nhiễu, Tỷ lệ nhiễu %, Silhouette].
        4. Trả về DataFrame kết quả để người dùng chọn cấu hình tối ưu.
        """
        pass

    def fit_model(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
        """
        Ý LÀM:
        1. Khởi tạo DBSCAN với cặp tham số (eps, min_samples) tốt nhất.
        2. Gọi fit(X).
        3. Trả về đối tượng mô hình DBSCAN đã huấn luyện.
        """
        pass

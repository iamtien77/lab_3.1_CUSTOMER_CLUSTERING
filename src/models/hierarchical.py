"""
=============================================================================
MODULE: PHÂN CỤM PHÂN CẤP (HIERARCHICAL / AGGLOMERATIVE CLUSTERING)
=============================================================================
Mục đích (Đáp ứng Bước 3 trong đề bài Practice Exercise 3):
- Xây dựng ma trận liên kết (Linkage Matrix) bằng các phương pháp: Ward, Complete, Average.
- Hỗ trợ vẽ biểu đồ nhánh phân cấp (Dendrogram) để xác định ngưỡng khoảng cách cắt tối ưu.
- Huấn luyện mô hình AgglomerativeClustering với số cụm K hoặc ngưỡng khoảng cách (distance threshold).
=============================================================================
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist


class HierarchicalTrainer:
    """
    Lớp quản lý phân cụm phân cấp từ dưới lên (Agglomerative Hierarchical Clustering).
    """

    def __init__(self, linkage_method: str = "ward"):
        """
        Ý LÀM:
        1. Nhận vào linkage_method:
           - 'ward': Giảm thiểu phương sai nội cụm (phổ biến nhất, tương đồng K-Means).
           - 'complete': Khoảng cách lớn nhất giữa hai điểm thuộc hai cụm.
           - 'average': Khoảng cách trung bình giữa tất cả các cặp điểm.
        2. Lưu phương pháp liên kết đã chọn.
        """
        pass

    def compute_linkage_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Ý LÀM:
        1. Sử dụng hàm scipy.cluster.hierarchy.linkage(X, method=self.linkage_method).
        2. Tính toán ma trận liên kết phân cấp chứa thông tin các cụm được gộp theo từng bước
           cùng khoảng cách giữa chúng.
        3. Tính hệ số tương quan Cophenetic (Cophenetic Correlation Coefficient) bằng cách so khớp
           khoảng cách trên cây phân cấp với ma trận khoảng cách gốc pdist(X) để đánh giá độ bảo toàn hình học.
        4. Trả về ma trận liên kết Z.
        """
        pass

    def fit_model(self, X: np.ndarray, n_clusters: int = 5) -> AgglomerativeClustering:
        """
        Ý LÀM:
        1. Khởi tạo AgglomerativeClustering(n_clusters=n_clusters, linkage=self.linkage_method).
        2. Gọi fit_predict(X) để học cấu trúc và sinh nhãn phân cụm cho từng mẫu dữ liệu.
        3. Lưu trữ nhãn cụm (labels_).
        4. Trả về đối tượng mô hình đã huấn luyện.
        """
        pass

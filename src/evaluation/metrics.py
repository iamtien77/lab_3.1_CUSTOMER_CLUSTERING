"""
=============================================================================
MODULE: ĐÁNH GIÁ CHẤT LƯỢNG PHÂN CỤM (CLUSTERING METRICS & BENCHMARK)
=============================================================================
Mục đích (Đáp ứng Bước 4 trong đề bài Practice Exercise 3):
- Tính toán toàn diện các chỉ số nội tại (Internal Clustering Metrics):
  * Hệ số Silhouette Score (và tỷ lệ điểm có Silhouette âm).
  * Chỉ số Davies-Bouldin Index (thấp hơn là tốt hơn).
  * Chỉ số Calinski-Harabasz Score (cao hơn là tốt hơn).
  * Chỉ số Dunn Index.
  * Độ cân bằng phân bổ kích thước giữa các cụm (Cluster Balance).
- Tạo bảng so sánh đối đầu toàn diện giữa cả 4 thuật toán:
  K-Means, Hierarchical Clustering, DBSCAN, và K-Medoids.
=============================================================================
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Ý LÀM (Chỉ số Dunn Index):
    1. Tính khoảng cách nhỏ nhất giữa hai điểm thuộc hai cụm khác nhau (Inter-cluster separation).
    2. Tính đường kính lớn nhất trong một cụm (Intra-cluster maximum diameter).
    3. Dunn Index = Khoảng cách liên cụm nhỏ nhất / Đường kính nội cụm lớn nhất.
    4. Giá trị Dunn Index càng cao chứng tỏ các cụm càng nén chặt và tách rời nhau.
    """
    pass


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Ý LÀM:
    1. Loại bỏ các điểm nhiễu (nhãn == -1 trong DBSCAN nếu có) khi tính toán các chỉ số phân tách.
    2. Đếm số cụm hợp lệ (n_clusters = số nhãn phân biệt > -1).
    3. Tính Silhouette Score:
       - Tính silhouette trung bình toàn bộ tập.
       - Tính silhouette từng mẫu (silhouette_samples) và đếm tỷ lệ mẫu bị âm (< 0, tức có nguy cơ bị gán nhầm cụm).
    4. Tính Davies-Bouldin Index (đo độ trùng lặp hình học giữa các cụm, giá trị càng nhỏ càng tốt).
    5. Tính Calinski-Harabasz Score (tỉ lệ phương sai giữa các cụm / trong nội bộ cụm, càng lớn càng tốt).
    6. Tính độ cân bằng kích thước cụm: tỷ lệ cụm nhỏ nhất so với cụm lớn nhất.
    7. Trả về Dictionary tổng hợp tất cả các chỉ số trên.
    """
    pass


def compare_all_models(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Ý LÀM:
    1. Nhận danh sách kết quả đánh giá của 4 thuật toán (K-Means, Hierarchical, DBSCAN, K-Medoids).
    2. Chuyển đổi thành DataFrame dạng bảng so sánh đối chiếu:
       [Thuật toán, Số cụm K, Silhouette Score, Negative Silhouette %, Davies-Bouldin, Calinski-Harabasz, Điểm nhiễu].
    3. Xếp hạng thuật toán có hiệu năng phân tách và cân bằng tốt nhất cho bài toán.
    4. Lưu kết quả ra file results/metrics/model_comparison.csv.
    5. Trả về DataFrame bảng so sánh.
    """
    pass

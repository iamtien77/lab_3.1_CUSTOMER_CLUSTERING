"""
=============================================================================
MODULE: TRỰC QUAN HÓA KẾT QUẢ PHÂN CỤM (CLUSTERING VISUALIZATION)
=============================================================================
Mục đích (Đáp ứng Bước 4 trong đề bài Practice Exercise 3):
- Trực quan hóa các cụm khách hàng trên đồ thị phân tán 2D (Scatter Plots)
  và không gian 3D (Age x Income x Spending Score).
- Vẽ biểu đồ nhánh cây phân cấp (Dendrograms) với đường ngưỡng cắt.
- Vẽ đồ thị Elbow (Inertia vs K) và đường cong Silhouette vs K.
- Vẽ biểu đồ phân tích Silhouette chi tiết (Silhouette Analysis Plot) cho từng cụm.
=============================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from src.config import (
    INCOME_COLUMN,
    SPENDING_COLUMN,
    AGE_COLUMN,
    CLUSTER_PALETTE,
    STYLE,
    save_figure,
)


def plot_elbow_and_silhouette(k_search_df: pd.DataFrame):
    """
    Ý LÀM:
    1. Tạo 2 khung biểu đồ phụ (subplot 1x2):
       - Khung trái: Đồ thị WCSS (Inertia) theo số cụm K -> Tìm điểm gãy (Khuỷu tay / Elbow point).
       - Khung phải: Đồ thị Silhouette Score theo số cụm K -> Tìm điểm đạt giá trị cực đại.
    2. Đánh dấu điểm K tối ưu bằng điểm tròn nổi bật và đường nét đứt.
    3. Lưu biểu đồ vào results/figures/01_elbow_and_silhouette.png.
    """
    pass


def plot_clusters_2d(
    df: pd.DataFrame,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    x_col: str = INCOME_COLUMN,
    y_col: str = SPENDING_COLUMN,
    title: str = "Phân Cụm Khách Hàng (2D Space)",
    filename: str = "02_clusters_2d.png",
):
    """
    Ý LÀM:
    1. Vẽ biểu đồ tán xạ 2 chiều của 2 đặc trưng quan trọng nhất (Annual Income vs Spending Score).
    2. Tô màu các điểm dữ liệu tương ứng với từng Cụm (Cluster label) theo bảng màu CLUSTER_PALETTE.
    3. Nếu có centroids (tâm cụm): vẽ các dấu sao hoặc dấu X to màu đỏ/vàng làm nổi bật tọa độ trung tâm.
    4. Thêm chú thích (Legend), tên trục, tiêu đề rõ ràng.
    5. Lưu file vào results/figures/{filename}.
    """
    pass


def plot_clusters_3d(
    df: pd.DataFrame,
    labels: np.ndarray,
    title: str = "Phân Cụm Khách Hàng Không Gian 3 Chiều",
    filename: str = "03_clusters_3d.png",
):
    """
    Ý LÀM:
    1. Khởi tạo trục tọa độ 3D (fig.add_subplot(projection='3d')).
    2. Đặt 3 trục lần lượt là: Trục X = Tuổi (Age), Trục Y = Thu Nhập (Annual Income), Trục Z = Điểm Chi Tiêu (Spending Score).
    3. Vẽ tán xạ 3D với màu sắc tương ứng theo từng cụm.
    4. Thiết lập góc nhìn quan sát (view_init(elev=20, azim=45)) để thấy rõ độ phân tách trong không gian 3 chiều.
    5. Lưu file vào results/figures/{filename}.
    """
    pass


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    cut_threshold: Optional[float] = None,
    filename: str = "04_hierarchical_dendrogram.png",
):
    """
    Ý LÀM:
    1. Gọi hàm scipy.cluster.hierarchy.dendrogram() trên ma trận liên kết.
    2. Cấu hình màu sắc nhánh cây trực quan.
    3. Nếu có cut_threshold (ngưỡng cắt): vẽ một đường nét đứt màu đỏ nằm ngang
       thể hiện vị trí cắt cây phân cấp để chia thành số cụm mong muốn.
    4. Lưu file vào results/figures/{filename}.
    """
    pass


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    optimal_k: int = 5,
    filename: str = "05_silhouette_analysis.png",
):
    """
    Ý LÀM:
    1. Tính toán silhouette_samples cho từng điểm dữ liệu.
    2. Sắp xếp hệ số silhouette trong từng cụm và vẽ biểu đồ hình lưỡi dao (Silhouette Knife Plot).
    3. Vẽ đường nét đứt biểu diễn điểm Silhouette trung bình của toàn tập dữ liệu.
    4. Đánh giá trực quan:
       - Các cụm có độ dày đồng đều không?
       - Có cụm nào có nhiều điểm bị âm (rơi sang bên trái đường 0) không?
    5. Lưu file vào results/figures/{filename}.
    """
    pass

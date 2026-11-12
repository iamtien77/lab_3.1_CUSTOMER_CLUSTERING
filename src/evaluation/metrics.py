"""
=============================================================================
MODULE: ĐÁNH GIÁ CHẤT LƯỢNG PHÂN CỤM (CLUSTERING METRICS & BENCHMARK)
=============================================================================
Mục đích (Đáp ứng Bước 4 trong đề bài Practice Exercise 3):
- Tính toán toàn diện các chỉ số nội tại (Internal Clustering Metrics):
  * Hệ số Silhouette Score (chính) và tỷ lệ điểm có Silhouette âm.
  * Chỉ số Davies-Bouldin Index (thấp hơn là tốt hơn).
  * Chỉ số Calinski-Harabasz Score (cao hơn là tốt hơn).
  * Chỉ số Dunn Index và độ cân bằng kích thước giữa các cụm.
- Tạo bảng so sánh đối chiếu giữa các thuật toán phân cụm.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from src.config import METRICS_DIR


def _filter_non_noise(feature_matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Lọc bỏ các điểm nhiễu (nhãn == -1) trước khi đánh giá cấu trúc phân tách cụm."""
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    label_array = np.asarray(labels)
    if len(matrix) != len(label_array):
        raise ValueError(f"Số lượng mẫu ({len(matrix)}) không khớp số lượng nhãn ({len(label_array)}).")

    valid_mask = label_array != -1
    noise_count = int(np.sum(~valid_mask))
    noise_ratio_pct = round(noise_count / max(len(label_array), 1) * 100.0, 2)
    return matrix[valid_mask], label_array[valid_mask], noise_count, noise_ratio_pct


def compute_dunn_index(feature_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Tính chỉ số Dunn Index = (Khoảng cách liên cụm nhỏ nhất) / (Đường kính nội cụm lớn nhất).
    Giá trị càng lớn chứng tỏ các cụm càng tách rời và nén chặt.
    """
    valid_matrix, valid_labels, _, _ = _filter_non_noise(feature_matrix, labels)
    unique_clusters = np.unique(valid_labels)
    if len(unique_clusters) < 2 or len(unique_clusters) >= len(valid_matrix):
        return float("nan")

    max_diameter = 0.0
    min_intercluster_dist = float("inf")
    cluster_points_list = [valid_matrix[valid_labels == cluster_id] for cluster_id in unique_clusters]

    for idx_i, cluster_i in enumerate(cluster_points_list):
        if len(cluster_i) >= 2:
            diameter_i = float(np.max(pdist(cluster_i, metric="euclidean")))
            if diameter_i > max_diameter:
                max_diameter = diameter_i

        for idx_j in range(idx_i + 1, len(cluster_points_list)):
            min_pair_dist = float(np.min(cdist(cluster_i, cluster_points_list[idx_j], metric="euclidean")))
            if min_pair_dist < min_intercluster_dist:
                min_intercluster_dist = min_pair_dist

    if max_diameter <= 0.0 or not np.isfinite(min_intercluster_dist):
        return float("nan")
    return round(min_intercluster_dist / max_diameter, 4)


def evaluate_clustering(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    model_name: str = "Model",
    compute_dunn: bool = True,
) -> Dict[str, Any]:
    """
    Đánh giá chất lượng phân cụm bằng các chỉ số nội tại với kiểm tra điều kiện biên chặt chẽ.
    Không gán giả 0 nếu số cụm hợp lệ vi phạm điều kiện 2 <= n_clusters <= n_samples - 1.
    """
    valid_matrix, valid_labels, noise_count, noise_ratio_pct = _filter_non_noise(feature_matrix, labels)
    unique_clusters, cluster_sizes = np.unique(valid_labels, return_counts=True)
    n_clusters = int(len(unique_clusters))
    n_valid_samples = len(valid_matrix)

    cluster_balance_ratio = (
        round(float(np.min(cluster_sizes) / np.max(cluster_sizes)), 4)
        if n_clusters >= 2
        else float("nan")
    )

    can_compute_metrics = 2 <= n_clusters <= (n_valid_samples - 1)
    if not can_compute_metrics:
        print(
            f" [CẢNH BÁO ĐÁNH GIÁ - {model_name}] Không đủ điều kiện tính chỉ số phân tách "
            f"(số cụm hợp lệ={n_clusters}, số mẫu={n_valid_samples})."
        )
        return {
            "model_name": model_name,
            "n_clusters": n_clusters,
            "n_valid_samples": n_valid_samples,
            "noise_count": noise_count,
            "noise_ratio_pct": noise_ratio_pct,
            "silhouette": float("nan"),
            "negative_silhouette_pct": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "dunn_index": float("nan"),
            "cluster_balance_ratio": cluster_balance_ratio,
        }

    avg_silhouette = float(silhouette_score(valid_matrix, valid_labels))
    sample_silhouettes = silhouette_samples(valid_matrix, valid_labels)
    negative_silhouette_pct = round(float(np.mean(sample_silhouettes < 0.0) * 100.0), 2)
    davies_bouldin = float(davies_bouldin_score(valid_matrix, valid_labels))
    calinski_harabasz = float(calinski_harabasz_score(valid_matrix, valid_labels))
    dunn_index = compute_dunn_index(valid_matrix, valid_labels) if compute_dunn else float("nan")

    return {
        "model_name": model_name,
        "n_clusters": n_clusters,
        "n_valid_samples": n_valid_samples,
        "noise_count": noise_count,
        "noise_ratio_pct": noise_ratio_pct,
        "silhouette": round(avg_silhouette, 4),
        "negative_silhouette_pct": negative_silhouette_pct,
        "davies_bouldin": round(davies_bouldin, 4),
        "calinski_harabasz": round(calinski_harabasz, 2),
        "dunn_index": dunn_index,
        "cluster_balance_ratio": cluster_balance_ratio,
    }


def compare_all_models(
    results_list: List[Dict[str, Any]],
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Tổng hợp bảng đối chiếu các thuật toán phân cụm, xếp hạng theo Silhouette Score và lưu CSV.
    """
    if not results_list:
        raise ValueError("Danh sách kết quả đánh giá đang rỗng, không thể lập bảng so sánh.")

    comparison_df = pd.DataFrame(results_list).rename(
        columns={
            "model_name": "Thuật toán",
            "n_clusters": "Số cụm K",
            "silhouette": "Silhouette Score",
            "negative_silhouette_pct": "Silhouette âm (%)",
            "davies_bouldin": "Davies-Bouldin",
            "calinski_harabasz": "Calinski-Harabasz",
            "dunn_index": "Dunn Index",
            "noise_count": "Điểm nhiễu",
            "noise_ratio_pct": "Tỷ lệ nhiễu (%)",
            "cluster_balance_ratio": "Độ cân bằng cụm",
        }
    )

    comparison_df = comparison_df.sort_values(
        by="Silhouette Score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    target_path = Path(output_path) if output_path is not None else METRICS_DIR / "model_comparison.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(target_path, index=False)
    print(f" [ĐÃ LƯU SO SÁNH MÔ HÌNH] -> {target_path}")
    return comparison_df

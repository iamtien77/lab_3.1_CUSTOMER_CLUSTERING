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


def _filter_non_noise(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Lọc bỏ các điểm nhiễu (nhãn == -1) trước khi đánh giá cấu trúc phân tách cụm."""
    ma_tran = np.asarray(X, dtype=np.float64)
    nhan_arr = np.asarray(labels)
    if len(ma_tran) != len(nhan_arr):
        raise ValueError(f"Số lượng mẫu X ({len(ma_tran)}) không khớp số lượng nhãn ({len(nhan_arr)}).")

    mask_hop_le = nhan_arr != -1
    so_nhieu = int(np.sum(~mask_hop_le))
    ty_le_nhieu = round(so_nhieu / max(len(nhan_arr), 1) * 100.0, 2)
    return ma_tran[mask_hop_le], nhan_arr[mask_hop_le], so_nhieu, ty_le_nhieu


def compute_dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Tính chỉ số Dunn Index = (Khoảng cách liên cụm nhỏ nhất) / (Đường kính nội cụm lớn nhất).
    Giá trị càng lớn chứng tỏ các cụm càng tách rời và nén chặt.
    """
    ma_tran_hop_le, nhan_hop_le, _, _ = _filter_non_noise(X, labels)
    cum_duy_nhat = np.unique(nhan_hop_le)
    if len(cum_duy_nhat) < 2 or len(cum_duy_nhat) >= len(ma_tran_hop_le):
        return float("nan")

    duong_kinh_toi_da = 0.0
    khoang_cach_lien_cum_min = float("inf")
    tap_cum = [ma_tran_hop_le[nhan_hop_le == c] for c in cum_duy_nhat]

    for idx_i, cum_i in enumerate(tap_cum):
        if len(cum_i) >= 2:
            duong_kinh_i = float(np.max(pdist(cum_i, metric="euclidean")))
            if duong_kinh_i > duong_kinh_toi_da:
                duong_kinh_toi_da = duong_kinh_i

        for idx_j in range(idx_i + 1, len(tap_cum)):
            kc_min_ij = float(np.min(cdist(cum_i, tap_cum[idx_j], metric="euclidean")))
            if kc_min_ij < khoang_cach_lien_cum_min:
                khoang_cach_lien_cum_min = kc_min_ij

    if duong_kinh_toi_da <= 0.0 or not np.isfinite(khoang_cach_lien_cum_min):
        return float("nan")
    return round(khoang_cach_lien_cum_min / duong_kinh_toi_da, 4)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str = "Model",
    compute_dunn: bool = True,
) -> Dict[str, Any]:
    """
    Đánh giá chất lượng phân cụm bằng các chỉ số nội tại với kiểm tra điều kiện biên chặt chẽ.
    Không gán giả 0 nếu số cụm hợp lệ vi phạm điều kiện 2 <= n_clusters <= n_samples - 1.
    """
    ma_tran_hop_le, nhan_hop_le, so_nhieu, ty_le_nhieu = _filter_non_noise(X, labels)
    cum_duy_nhat, dem_kich_thuoc = np.unique(nhan_hop_le, return_counts=True)
    so_cum = int(len(cum_duy_nhat))
    so_mau_hop_le = len(ma_tran_hop_le)

    ty_le_can_bang = (
        round(float(np.min(dem_kich_thuoc) / np.max(dem_kich_thuoc)), 4)
        if so_cum >= 2
        else float("nan")
    )

    hop_le_tinh_chi_so = 2 <= so_cum <= (so_mau_hop_le - 1)
    if not hop_le_tinh_chi_so:
        print(
            f" [CẢNH BÁO ĐÁNH GIÁ - {model_name}] Không đủ điều kiện tính chỉ số phân tách "
            f"(số cụm hợp lệ={so_cum}, số mẫu={so_mau_hop_le})."
        )
        return {
            "model_name": model_name,
            "n_clusters": so_cum,
            "n_valid_samples": so_mau_hop_le,
            "noise_count": so_nhieu,
            "noise_ratio_pct": ty_le_nhieu,
            "silhouette": float("nan"),
            "negative_silhouette_pct": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "dunn_index": float("nan"),
            "cluster_balance_ratio": ty_le_can_bang,
        }

    silhouette_trung_binh = float(silhouette_score(ma_tran_hop_le, nhan_hop_le))
    silhouette_tung_mau = silhouette_samples(ma_tran_hop_le, nhan_hop_le)
    ty_le_silhouette_am = round(float(np.mean(silhouette_tung_mau < 0.0) * 100.0), 2)
    davies_bouldin = float(davies_bouldin_score(ma_tran_hop_le, nhan_hop_le))
    calinski_harabasz = float(calinski_harabasz_score(ma_tran_hop_le, nhan_hop_le))
    dunn_index = compute_dunn_index(ma_tran_hop_le, nhan_hop_le) if compute_dunn else float("nan")

    return {
        "model_name": model_name,
        "n_clusters": so_cum,
        "n_valid_samples": so_mau_hop_le,
        "noise_count": so_nhieu,
        "noise_ratio_pct": ty_le_nhieu,
        "silhouette": round(silhouette_trung_binh, 4),
        "negative_silhouette_pct": ty_le_silhouette_am,
        "davies_bouldin": round(davies_bouldin, 4),
        "calinski_harabasz": round(calinski_harabasz, 2),
        "dunn_index": dunn_index,
        "cluster_balance_ratio": ty_le_can_bang,
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

    bang_doi_chieu = pd.DataFrame(results_list).rename(
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

    bang_doi_chieu = bang_doi_chieu.sort_values(
        by="Silhouette Score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    duong_dan_luu = Path(output_path) if output_path is not None else METRICS_DIR / "model_comparison.csv"
    duong_dan_luu.parent.mkdir(parents=True, exist_ok=True)
    bang_doi_chieu.to_csv(duong_dan_luu, index=False)
    print(f" [ĐÃ LƯU SO SÁNH MÔ HÌNH] -> {duong_dan_luu}")
    return bang_doi_chieu

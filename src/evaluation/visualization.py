"""
=============================================================================
MODULE: TRỰC QUAN HÓA DỮ LIỆU & KẾT QUẢ PHÂN CỤM (VISUALIZATION)
=============================================================================
Mục đích (Đáp ứng Bước 1, 2, 3 và 4 trong đề bài Practice Exercise 3):
- Trực quan hóa EDA: Histogram/KDE, Boxplot IQR, Scatter song biến và Heatmap tương quan.
- Trực quan hóa phân phối trước & sau chuẩn hóa đặc trưng.
- Trực quan hóa quá trình chọn tham số: Elbow + Silhouette, Dendrogram, Đồ thị k-NN DBSCAN.
- Trực quan hóa kết quả phân cụm 2D, 3D và Silhouette Knife Plot.
=============================================================================
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.config import (
    AGE_COLUMN,
    CLUSTER_PALETTE,
    INCOME_COLUMN,
    NUMERICAL_FEATURES,
    SPENDING_COLUMN,
    STYLE,
    save_figure,
)


_FEATURE_LABELS = {
    AGE_COLUMN: "Tuổi (năm)",
    INCOME_COLUMN: "Thu nhập hộ gia đình (USD/năm)",
    SPENDING_COLUMN: "Tổng chi tiêu (USD)",
}


def _axis_label(col_name: str) -> str:
    """Trả về nhãn trục biểu đồ kèm đơn vị rõ ràng theo chuẩn thiết kế."""
    return _FEATURE_LABELS.get(col_name, col_name)


def _finalize_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Lưu biểu đồ ra thư mục đích (hoặc FIGURES_DIR mặc định) và đóng Figure giải phóng RAM."""
    if output_dir is not None:
        saved_path = Path(output_dir) / filename
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(saved_path, dpi=200, bbox_inches="tight")
        print(f" [ĐÃ LƯU BIỂU ĐỒ] -> {saved_path}")
    else:
        saved_path = save_figure(filename)
    plt.close(fig)
    return saved_path


def plot_eda_distributions(
    df: pd.DataFrame,
    filename: str = "00a_eda_distributions.png",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Vẽ Histogram/KDE và Boxplot cho 3 đặc trưng số Age, Income, Total_Spending."""
    display_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    fig, axes = plt.subplots(2, len(display_cols), figsize=(5 * len(display_cols), 8))
    if len(display_cols) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, col in enumerate(display_cols):
        axis_label_text = _axis_label(col)
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0, i], color=STYLE["blue"])
        axes[0, i].set_title(f"Phân phối {axis_label_text}")
        axes[0, i].set_xlabel(axis_label_text)
        axes[0, i].set_ylabel("Tần suất (khách hàng)")
        axes[0, i].xaxis.set_major_locator(MaxNLocator(nbins=5))

        sns.boxplot(x=df[col].dropna(), ax=axes[1, i], color=STYLE["soft_blue"])
        axes[1, i].set_title(f"Boxplot phát hiện ngoại lai: {axis_label_text}")
        axes[1, i].set_xlabel(axis_label_text)
        axes[1, i].xaxis.set_major_locator(MaxNLocator(nbins=5))

        if col in (INCOME_COLUMN, SPENDING_COLUMN):
            currency_formatter = FuncFormatter(
                lambda val, _: (
                    f"{val / 1000:.1f}k".replace(".0k", "k")
                    if abs(val) >= 1000
                    else f"{val:g}"
                )
            )
            axes[0, i].xaxis.set_major_formatter(currency_formatter)
            axes[1, i].xaxis.set_major_formatter(currency_formatter)

    fig.suptitle("Khám Phá Phân Phối Và Ngoại Lai Các Đặc Trưng Khách Hàng", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, filename, output_dir)


def plot_eda_correlations(
    df: pd.DataFrame,
    filename: str = "00b_eda_correlations.png",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Vẽ Scatter song biến (Income/Age vs Total_Spending), Heatmap tương quan và phân bố Education."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].scatter(df[INCOME_COLUMN], df[SPENDING_COLUMN], alpha=0.6, color=STYLE["blue"], edgecolors="none")
    axes[0, 0].set_title(f"Tương quan Thu nhập và Tổng chi tiêu")
    axes[0, 0].set_xlabel(_axis_label(INCOME_COLUMN))
    axes[0, 0].set_ylabel(_axis_label(SPENDING_COLUMN))

    axes[0, 1].scatter(df[AGE_COLUMN], df[SPENDING_COLUMN], alpha=0.6, color=STYLE["secondary"], edgecolors="none")
    axes[0, 1].set_title(f"Tương quan Tuổi và Tổng chi tiêu")
    axes[0, 1].set_xlabel(_axis_label(AGE_COLUMN))
    axes[0, 1].set_ylabel(_axis_label(SPENDING_COLUMN))

    numeric_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=axes[1, 0], square=True)
    axes[1, 0].set_title("Ma trận hệ số tương quan Pearson")

    if "Education" in df.columns:
        education_counts = df["Education"].value_counts()
        axes[1, 1].bar(education_counts.index, education_counts.values, color=STYLE["primary"])
        axes[1, 1].set_title("Phân bố trình độ học vấn (Education)")
        axes[1, 1].set_ylabel("Số khách hàng")
        axes[1, 1].tick_params(axis="x", rotation=20)
    else:
        axes[1, 1].axis("off")

    fig.suptitle("Phân Tích Quan Hệ Song Biến Và Nhân Khẩu Học", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, filename, output_dir)


def plot_scaling_comparison(
    df_raw: pd.DataFrame,
    scaled_matrix: np.ndarray,
    feature_cols: List[str] = NUMERICAL_FEATURES,
    filename: str = "00c_scaling_comparison.png",
    output_dir: Optional[Path | str] = None,
    scaled_variants: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """So sánh trực quan phân phối dữ liệu trước và sau chuẩn hóa qua StandardScaler, MinMaxScaler và RobustScaler."""
    valid_cols = [col for col in feature_cols if col in df_raw.columns]
    standard_matrix = np.asarray(scaled_matrix, dtype=np.float64)
    if scaled_variants is None:
        variant_matrices: Dict[str, np.ndarray] = {
            "StandardScaler (Mean=0, Std=1)": standard_matrix,
        }
        if len(df_raw) >= 2 and valid_cols:
            raw_matrix = df_raw[valid_cols].to_numpy(dtype=np.float64)
            if np.all(np.isfinite(raw_matrix)):
                variant_matrices["MinMaxScaler (Đưa về [0, 1])"] = MinMaxScaler().fit_transform(raw_matrix)
                variant_matrices["RobustScaler (Median=0, IQR=1)"] = RobustScaler().fit_transform(raw_matrix)
    else:
        variant_matrices = scaled_variants

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    palette_colors = CLUSTER_PALETTE[:3]

    ax_raw = axes[0, 0]
    for idx, col in enumerate(valid_cols):
        color = palette_colors[idx % len(palette_colors)]
        sns.kdeplot(
            df_raw[col],
            ax=ax_raw,
            label=_axis_label(col),
            color=color,
            linewidth=2.0,
            warn_singular=False,
        )
    ax_raw.set_title("1. Trước Chuẩn Hóa (Thang đo gốc chênh lệch lớn)")
    ax_raw.set_xlabel("Giá trị gốc")
    ax_raw.set_ylabel("Mật độ (KDE)")
    ax_raw.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_raw.xaxis.set_major_formatter(
        FuncFormatter(
            lambda val, _: (
                f"{val / 1000:.1f}k".replace(".0k", "k")
                if abs(val) >= 1000
                else f"{val:g}"
            )
        )
    )
    ax_raw.grid(True, alpha=0.25)
    ax_raw.legend(loc="upper right", fontsize=8)
    if len(valid_cols) >= 2:
        ax_raw.text(
            0.03,
            0.78,
            "Age (18-74) bị nén sát mốc 0\ndo Income lớn gấp ~2.000 lần",
            transform=ax_raw.transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )

    subplot_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax_pos in subplot_axes:
        ax_pos.set_visible(False)

    x_axis_labels = [
        "Giá trị chuẩn hóa Z-score",
        "Giá trị chuẩn hóa [0, 1]",
        "Giá trị chuẩn hóa theo Median / IQR",
    ]
    for panel_idx, (method_title, method_matrix) in enumerate(list(variant_matrices.items())[:3]):
        ax = subplot_axes[panel_idx]
        ax.set_visible(True)
        method_array = np.asarray(method_matrix, dtype=np.float64)
        for col_idx, col in enumerate(valid_cols):
            if col_idx < method_array.shape[1]:
                color = palette_colors[col_idx % len(palette_colors)]
                sns.kdeplot(
                    method_array[:, col_idx],
                    ax=ax,
                    label=_axis_label(col),
                    color=color,
                    linewidth=2.0,
                    warn_singular=False,
                )
        ax.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_title(f"{panel_idx + 2}. Sau {method_title}")
        ax.set_xlabel(x_axis_labels[panel_idx] if panel_idx < len(x_axis_labels) else "Giá trị sau chuẩn hóa")
        ax.set_ylabel("Mật độ (KDE)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("So Sánh Thang Đo Gốc Và Ba Phương Pháp Chuẩn Hóa Đặc Trưng", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, filename, output_dir)


def plot_elbow_and_silhouette(
    k_search_df: pd.DataFrame,
    optimal_k: Optional[int] = None,
    filename: str = "01_elbow_and_silhouette.png",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Vẽ đồ thị Elbow (WCSS/Inertia) và đường cong Silhouette Score theo số cụm K."""
    if optimal_k is None and not k_search_df["Silhouette"].dropna().empty:
        optimal_k = int(k_search_df.loc[k_search_df["Silhouette"].idxmax(), "K"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(k_search_df["K"], k_search_df["Inertia"], marker="o", color=STYLE["blue"], linewidth=2)
    axes[0].set_title("Phương pháp Khuỷu tay (Elbow - Inertia)")
    axes[0].set_xlabel("Số cụm K")
    axes[0].set_ylabel("WCSS (Inertia)")

    axes[1].plot(k_search_df["K"], k_search_df["Silhouette"], marker="s", color=STYLE["green"], linewidth=2)
    axes[1].set_title("Hệ số Silhouette theo số cụm K")
    axes[1].set_xlabel("Số cụm K")
    axes[1].set_ylabel("Silhouette Score")

    if optimal_k is not None and optimal_k in set(k_search_df["K"]):
        optimal_row = k_search_df[k_search_df["K"] == optimal_k].iloc[0]
        for ax, metric_col in [(axes[0], "Inertia"), (axes[1], "Silhouette")]:
            ax.axvline(optimal_k, color=STYLE["red"], linestyle="--", alpha=0.8, label=f"K đề xuất = {optimal_k}")
            ax.scatter([optimal_k], [optimal_row[metric_col]], color=STYLE["red"], s=90, zorder=5)
            ax.legend()

    fig.suptitle("Khảo Sát Số Cụm Tối Ưu K-Means", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, filename, output_dir)


def plot_clusters_2d(
    df: pd.DataFrame,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    x_col: str = INCOME_COLUMN,
    y_col: str = SPENDING_COLUMN,
    title: str = "Phân Cụm Khách Hàng (Không gian 2D)",
    filename: str = "02_clusters_2d.png",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Vẽ biểu đồ phân tán 2D các cụm kèm tâm cụm (nếu truyền vào ở đơn vị gốc)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    label_array = np.asarray(labels)
    cluster_ids = sorted([c for c in np.unique(label_array) if c != -1])

    for i, cluster_id in enumerate(cluster_ids):
        mask = label_array == cluster_id
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], s=35, alpha=0.75, color=color, label=f"Cụm {cluster_id}")

    if -1 in label_array:
        noise_mask = label_array == -1
        ax.scatter(
            df.loc[noise_mask, x_col],
            df.loc[noise_mask, y_col],
            s=30,
            color="#64748B",
            marker="x",
            label="Nhiễu (-1)",
        )

    if centroids is not None and len(centroids) > 0:
        if centroids.shape[1] == len(NUMERICAL_FEATURES):
            x_col_idx = NUMERICAL_FEATURES.index(x_col) if x_col in NUMERICAL_FEATURES else 1
            y_col_idx = NUMERICAL_FEATURES.index(y_col) if y_col in NUMERICAL_FEATURES else 2
        else:
            x_col_idx, y_col_idx = 0, min(1, centroids.shape[1] - 1)
        ax.scatter(
            centroids[:, x_col_idx],
            centroids[:, y_col_idx],
            s=220,
            c=STYLE["red"],
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            label="Tâm cụm (Centroids)",
            zorder=10,
        )

    ax.set_title(title)
    ax.set_xlabel(_axis_label(x_col))
    ax.set_ylabel(_axis_label(y_col))
    ax.legend(loc="best")
    return _finalize_figure(fig, filename, output_dir)


def plot_clusters_3d(
    df: pd.DataFrame,
    labels: np.ndarray,
    title: str = "Phân Cụm Khách Hàng Không Gian 3 Chiều",
    filename: str = "03_clusters_3d.png",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Vẽ phân tán 3D trên không gian Age x Income x Total_Spending."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    label_array = np.asarray(labels)
    cluster_ids = sorted([c for c in np.unique(label_array) if c != -1])

    for i, cluster_id in enumerate(cluster_ids):
        mask = label_array == cluster_id
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        ax.scatter(
            df.loc[mask, AGE_COLUMN],
            df.loc[mask, INCOME_COLUMN],
            df.loc[mask, SPENDING_COLUMN],
            s=28,
            alpha=0.75,
            color=color,
            label=f"Cụm {cluster_id}",
        )

    ax.set_title(title)
    ax.set_xlabel(_axis_label(AGE_COLUMN))
    ax.set_ylabel(_axis_label(INCOME_COLUMN))
    ax.set_zlabel(_axis_label(SPENDING_COLUMN))
    ax.view_init(elev=20, azim=45)
    ax.legend(loc="best")
    return _finalize_figure(fig, filename, output_dir)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    cut_threshold: Optional[float] = None,
    filename: str = "04_hierarchical_dendrogram.png",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Vẽ biểu đồ nhánh phân cấp (Dendrogram) rút gọn 30 nhánh cuối kèm ngưỡng cắt."""
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90.0,
        leaf_font_size=9.0,
        show_contracted=True,
        ax=ax,
    )
    if cut_threshold is not None:
        ax.axhline(y=cut_threshold, color=STYLE["red"], linestyle="--", linewidth=1.8, label=f"Ngưỡng cắt = {cut_threshold:.2f}")
        ax.legend()

    ax.set_title("Biểu Đồ Cây Phân Cấp Khách Hàng (Hierarchical Dendrogram)")
    ax.set_xlabel("Nhánh cụm (số lượng điểm trong ngoặc)")
    ax.set_ylabel("Khoảng cách hợp nhất (Euclidean / Ward)")
    return _finalize_figure(fig, filename, output_dir)


def plot_silhouette_analysis(
    scaled_matrix: np.ndarray,
    labels: np.ndarray,
    optimal_k: Optional[int] = None,
    filename: str = "05_silhouette_analysis.png",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Vẽ biểu đồ lưỡi dao Silhouette (Silhouette Knife Plot) chi tiết theo từng cụm."""
    matrix = np.asarray(scaled_matrix, dtype=np.float64)
    label_array = np.asarray(labels)
    mask = label_array != -1
    valid_matrix, valid_labels = matrix[mask], label_array[mask]
    cluster_ids = sorted(np.unique(valid_labels))

    fig, ax = plt.subplots(figsize=(9, 6))
    if len(cluster_ids) < 2:
        ax.text(0.5, 0.5, "Không đủ cụm (>= 2) để vẽ Silhouette Plot", ha="center", va="center")
        return _finalize_figure(fig, filename, output_dir)

    avg_silhouette = float(silhouette_score(valid_matrix, valid_labels))
    sample_silhouettes = silhouette_samples(valid_matrix, valid_labels)
    n_clusters = len(cluster_ids) if optimal_k is None else optimal_k

    x_min = min(-0.24, float(sample_silhouettes.min()) - 0.14)
    x_max = min(1.05, max(0.85, float(sample_silhouettes.max()) + 0.08))
    y_lower = 10
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_values = np.sort(sample_silhouettes[valid_labels == cluster_id])
        y_upper = y_lower + len(cluster_values)
        color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)] if idx < len(CLUSTER_PALETTE) else cm.nipy_spectral(float(idx) / max(n_clusters, 1))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, facecolor=color, edgecolor=color, alpha=0.75)
        ax.text(
            x_min + 0.015,
            y_lower + 0.5 * len(cluster_values),
            f"Cụm {cluster_id}\n(n={len(cluster_values):,})",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
        y_lower = y_upper + 10

    ax.axvline(x=avg_silhouette, color=STYLE["red"], linestyle="--", linewidth=1.8, label=f"Silhouette TB = {avg_silhouette:.4f}")
    ax.axvline(x=0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_xlim(x_min, x_max)
    ax.set_title(f"Phân Tích Silhouette Chi Tiết Cho {len(cluster_ids)} Cụm")
    ax.set_xlabel("Giá trị hệ số Silhouette")
    ax.set_ylabel("Phân bố các mẫu theo từng cụm")
    ax.set_yticks([])
    ax.legend(loc="best")
    return _finalize_figure(fig, filename, output_dir)


def plot_dbscan_k_distance(
    k_distances: np.ndarray,
    k: int = 4,
    filename: str = "06_dbscan_k_distance.png",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """Vẽ đường cong khoảng cách k-NN sắp xếp tăng dần để chọn bán kính eps cho DBSCAN."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(k_distances)), k_distances, color=STYLE["purple"], linewidth=2)
    ax.set_title(f"Đồ Thị Khoảng Cách Láng Giềng Thứ {k} (Gợi Ý Tham Số eps Cho DBSCAN)")
    ax.set_xlabel("Chỉ số điểm dữ liệu (đã sắp xếp)")
    ax.set_ylabel(f"Khoảng cách tới điểm gần thứ {k}")
    return _finalize_figure(fig, filename, output_dir)

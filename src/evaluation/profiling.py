"""
=============================================================================
MODULE: PHÂN TÍCH HỒ SƠ PHÂN KHÚC KHÁCH HÀNG (CUSTOMER PROFILING & PERSONAS)
=============================================================================
Mục đích:
- Phân tích thống kê đặc trưng nhân khẩu học (Tuổi, Học vấn, Hôn nhân)
  và hành vi tài chính/chi tiêu (Thu nhập, Tổng chi tiêu) theo từng cụm.
- Định danh Persona động theo dữ liệu thực tế cho số cụm K bất kỳ.
- Đề xuất chiến lược tiếp thị (Marketing/CRM) phù hợp cho từng phân khúc.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    AGE_COLUMN,
    INCOME_COLUMN,
    METRICS_DIR,
    SPENDING_COLUMN,
    SPENDING_COMPONENT_COLUMNS,
)


def _mode_or_na(series: pd.Series) -> str:
    """Lấy giá trị xuất hiện nhiều nhất của cột phân loại kèm tỷ lệ %."""
    valid_values = series.dropna()
    if valid_values.empty:
        return "N/A"
    counts = valid_values.value_counts()
    top_value = counts.index[0]
    top_pct = counts.iloc[0] / len(valid_values) * 100.0
    return f"{top_value} ({top_pct:.1f}%)"


def create_cluster_profiles(
    df: pd.DataFrame,
    labels: pd.Series | np.ndarray | list,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Tổng hợp bảng thống kê mô tả chi tiết cho từng cụm và lưu ra tệp CSV.
    """
    label_array = np.asarray(labels)
    if len(df) != len(label_array):
        raise ValueError(f"Số dòng DataFrame ({len(df)}) không khớp số nhãn ({len(label_array)}).")

    labeled_df = df.copy()
    labeled_df["Cluster"] = label_array
    total_count = max(len(labeled_df), 1)

    profile_rows = []
    # Sắp xếp cụm thường theo thứ tự tăng dần và đặt nhiễu (-1) ở cuối nếu có
    cluster_ids = sorted([cluster_val for cluster_val in labeled_df["Cluster"].unique() if cluster_val != -1])
    if -1 in labeled_df["Cluster"].values:
        cluster_ids.append(-1)

    for cluster_id in cluster_ids:
        cluster_df = labeled_df[labeled_df["Cluster"] == cluster_id]
        row_dict = {
            "Cluster": int(cluster_id),
            "Số lượng": int(len(cluster_df)),
            "Tỷ lệ (%)": round(len(cluster_df) / total_count * 100.0, 2),
            "Tuổi TB": round(float(cluster_df[AGE_COLUMN].mean()), 2),
            "Tuổi Trung vị": round(float(cluster_df[AGE_COLUMN].median()), 2),
            "Thu nhập TB": round(float(cluster_df[INCOME_COLUMN].mean()), 2),
            "Thu nhập Trung vị": round(float(cluster_df[INCOME_COLUMN].median()), 2),
            "Chi tiêu TB": round(float(cluster_df[SPENDING_COLUMN].mean()), 2),
            "Chi tiêu Trung vị": round(float(cluster_df[SPENDING_COLUMN].median()), 2),
        }
        if "Education" in cluster_df.columns:
            row_dict["Học vấn phổ biến"] = _mode_or_na(cluster_df["Education"])
        if "Marital_Status" in cluster_df.columns:
            row_dict["Hôn nhân phổ biến"] = _mode_or_na(cluster_df["Marital_Status"])
        if all(col in cluster_df.columns for col in SPENDING_COMPONENT_COLUMNS):
            category_means = cluster_df[SPENDING_COMPONENT_COLUMNS].mean()
            top_category = str(category_means.idxmax())
            row_dict["Danh mục chi chính"] = f"{top_category} (TB: {category_means[top_category]:,.1f})"
        profile_rows.append(row_dict)

    profile_df = pd.DataFrame(profile_rows)
    target_path = Path(output_path) if output_path is not None else METRICS_DIR / "cluster_profiles.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(target_path, index=False)
    print(f" [HỒ SƠ PHÂN KHÚC] Đã lập bảng thống kê {len(profile_df)} cụm và lưu tại: {target_path}")
    return profile_df


def _classify_level(value: float, baseline: float, margin: float = 0.12) -> str:
    """Phân ngưỡng Cao / Trung bình / Thấp tương đối so với mốc trung bình toàn tập."""
    if baseline <= 0:
        return "Trung bình"
    if value >= baseline * (1.0 + margin):
        return "Cao"
    if value <= baseline * (1.0 - margin):
        return "Thấp"
    return "Trung bình"


_GROUP_STRATEGIES = {
    ("Cao", "Cao"): (
        "Khách hàng VIP / Giá trị cao",
        "Ưu tiên chăm sóc hội viên VIP, truyền thông các danh mục chi tiêu chủ lực và cá nhân hóa ưu đãi duy trì mức chi tiêu cao.",
    ),
    ("Cao", "Thấp"): (
        "Khách hàng Cẩn trọng / Tiềm năng chưa khai thác",
        "Tập trung truyền thông giá trị bền vững, bảo hành chất lượng và ưu đãi dùng thử danh mục sản phẩm mới để kích cầu.",
    ),
    ("Cao", "Trung bình"): (
        "Khách hàng Khá giả / Chi tiêu ổn định",
        "Gợi ý bán chéo (cross-sell) danh mục cao cấp và ưu đãi theo giá trị hóa đơn để nâng mức chi tiêu.",
    ),
    ("Thấp", "Cao"): (
        "Khách hàng Nhạy xu hướng / Chi tiêu mạnh",
        "Đẩy mạnh Flash Sale, mã giảm giá ngắn hạn và combo trải nghiệm bắt trend.",
    ),
    ("Thấp", "Thấp"): (
        "Khách hàng Tiết kiệm / Nhạy cảm giá",
        "Áp dụng combo thiết yếu giá tốt, chương trình tích điểm đổi quà và khuyến mãi theo mùa.",
    ),
    ("Thấp", "Trung bình"): (
        "Khách hàng Bình dân / Mua sắm vừa phải",
        "Duy trì tương tác bằng ưu đãi mua theo gói tiết kiệm và miễn phí vận chuyển đơn nhỏ.",
    ),
}


def generate_customer_personas(profile_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Định danh Persona động dựa trên đặc trưng trung bình của từng cụm so với toàn tập dữ liệu.
    Hoạt động tổng quát cho mọi giá trị K tối ưu tìm được từ thực nghiệm.
    """
    if profile_df.empty:
        return []

    valid_clusters_df = profile_df[profile_df["Cluster"] != -1]
    if valid_clusters_df.empty:
        valid_clusters_df = profile_df

    total_size = max(float(valid_clusters_df["Số lượng"].sum()), 1.0)
    overall_avg_age = float((valid_clusters_df["Tuổi TB"] * valid_clusters_df["Số lượng"]).sum() / total_size)
    overall_avg_income = float((valid_clusters_df["Thu nhập TB"] * valid_clusters_df["Số lượng"]).sum() / total_size)
    overall_avg_spending = float((valid_clusters_df["Chi tiêu TB"] * valid_clusters_df["Số lượng"]).sum() / total_size)

    name_counts: Dict[str, int] = {}
    preliminary_results = []
    for _, row in profile_df.iterrows():
        cluster_id = int(row["Cluster"])
        if cluster_id == -1:
            persona_name = "Khách hàng Ngoại lai / Hành vi đặc biệt"
            marketing_strategy = "Kiểm tra thủ công lịch sử giao dịch để loại trừ dữ liệu bất thường hoặc phục vụ riêng."
        else:
            income_level = _classify_level(float(row["Thu nhập TB"]), overall_avg_income)
            spending_level = _classify_level(float(row["Chi tiêu TB"]), overall_avg_spending)
            base_name, marketing_strategy = _GROUP_STRATEGIES.get(
                (income_level, spending_level),
                (
                    "Khách hàng Tiêu chuẩn / Trung bình",
                    "Duy trì tần suất mua sắm đều đặn qua email marketing, voucher sinh nhật và tích điểm thành viên.",
                ),
            )
            age_tag = "Trẻ/Trung niên trẻ" if float(row["Tuổi TB"]) < overall_avg_age - 2.0 else (
                "Trung niên/Cao tuổi" if float(row["Tuổi TB"]) > overall_avg_age + 2.0 else "Độ tuổi phổ thông"
            )
            persona_name = f"{base_name} ({age_tag})"

        name_counts[persona_name] = name_counts.get(persona_name, 0) + 1
        preliminary_results.append((cluster_id, row, persona_name, marketing_strategy))

    # Gắn chỉ số phân biệt nếu có từ 2 cụm trùng tên Persona
    name_indices: Dict[str, int] = {}
    personas: List[Dict[str, Any]] = []
    for cluster_id, row, persona_name, marketing_strategy in preliminary_results:
        final_persona_name = persona_name
        if name_counts[persona_name] > 1:
            name_indices[persona_name] = name_indices.get(persona_name, 0) + 1
            final_persona_name = f"{persona_name} #{name_indices[persona_name]}"

        personas.append(
            {
                "cluster_id": cluster_id,
                "persona_name": final_persona_name,
                "size": int(row["Số lượng"]),
                "pct": float(row["Tỷ lệ (%)"]),
                "avg_age": float(row["Tuổi TB"]),
                "avg_income": float(row["Thu nhập TB"]),
                "avg_spending": float(row["Chi tiêu TB"]),
                "marketing_strategy": marketing_strategy,
            }
        )

    print(f" [ĐỊNH DANH PERSONA] Đã xây dựng chân dung và chiến lược cho {len(personas)} phân khúc.")
    return personas

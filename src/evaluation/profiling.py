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
    hop_le = series.dropna()
    if hop_le.empty:
        return "N/A"
    gia_tri = hop_le.value_counts()
    ten_pho_bien = gia_tri.index[0]
    ty_le = gia_tri.iloc[0] / len(hop_le) * 100.0
    return f"{ten_pho_bien} ({ty_le:.1f}%)"


def create_cluster_profiles(
    df: pd.DataFrame,
    labels: pd.Series | np.ndarray | list,
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Tổng hợp bảng thống kê mô tả chi tiết cho từng cụm và lưu ra tệp CSV.
    """
    nhan_arr = np.asarray(labels)
    if len(df) != len(nhan_arr):
        raise ValueError(f"Số dòng DataFrame ({len(df)}) không khớp số nhãn ({len(nhan_arr)}).")

    df_gan_nhan = df.copy()
    df_gan_nhan["Cluster"] = nhan_arr
    tong_so = max(len(df_gan_nhan), 1)

    cac_dong = []
    # Sắp xếp cụm thường theo thứ tự tăng dần và đặt nhiễu (-1) ở cuối nếu có
    cac_cum = sorted([c for c in df_gan_nhan["Cluster"].unique() if c != -1])
    if -1 in df_gan_nhan["Cluster"].values:
        cac_cum.append(-1)

    for cum_id in cac_cum:
        nhom = df_gan_nhan[df_gan_nhan["Cluster"] == cum_id]
        dong = {
            "Cluster": int(cum_id),
            "Số lượng": int(len(nhom)),
            "Tỷ lệ (%)": round(len(nhom) / tong_so * 100.0, 2),
            "Tuổi TB": round(float(nhom[AGE_COLUMN].mean()), 2),
            "Tuổi Trung vị": round(float(nhom[AGE_COLUMN].median()), 2),
            "Thu nhập TB": round(float(nhom[INCOME_COLUMN].mean()), 2),
            "Thu nhập Trung vị": round(float(nhom[INCOME_COLUMN].median()), 2),
            "Chi tiêu TB": round(float(nhom[SPENDING_COLUMN].mean()), 2),
            "Chi tiêu Trung vị": round(float(nhom[SPENDING_COLUMN].median()), 2),
        }
        if "Education" in nhom.columns:
            dong["Học vấn phổ biến"] = _mode_or_na(nhom["Education"])
        if "Marital_Status" in nhom.columns:
            dong["Hôn nhân phổ biến"] = _mode_or_na(nhom["Marital_Status"])
        if all(col in nhom.columns for col in SPENDING_COMPONENT_COLUMNS):
            trung_binh_danh_muc = nhom[SPENDING_COMPONENT_COLUMNS].mean()
            ten_top = str(trung_binh_danh_muc.idxmax())
            dong["Danh mục chi chính"] = f"{ten_top} (TB: {trung_binh_danh_muc[ten_top]:,.1f})"
        cac_dong.append(dong)

    profile_df = pd.DataFrame(cac_dong)
    duong_dan_luu = Path(output_path) if output_path is not None else METRICS_DIR / "cluster_profiles.csv"
    duong_dan_luu.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(duong_dan_luu, index=False)
    print(f" [HỒ SƠ PHÂN KHÚC] Đã lập bảng thống kê {len(profile_df)} cụm và lưu tại: {duong_dan_luu}")
    return profile_df


def _phan_muc(gia_tri: float, moc_chuan: float, bien_do: float = 0.12) -> str:
    """Phân ngưỡng Cao / Trung bình / Thấp tương đối so với mốc trung bình toàn tập."""
    if moc_chuan <= 0:
        return "Trung bình"
    if gia_tri >= moc_chuan * (1.0 + bien_do):
        return "Cao"
    if gia_tri <= moc_chuan * (1.0 - bien_do):
        return "Thấp"
    return "Trung bình"


_CHIEN_LUOC_THEO_NHOM = {
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

    cum_hop_le = profile_df[profile_df["Cluster"] != -1]
    if cum_hop_le.empty:
        cum_hop_le = profile_df

    tong_so = max(float(cum_hop_le["Số lượng"].sum()), 1.0)
    tb_tuoi_chung = float((cum_hop_le["Tuổi TB"] * cum_hop_le["Số lượng"]).sum() / tong_so)
    tb_thu_nhap_chung = float((cum_hop_le["Thu nhập TB"] * cum_hop_le["Số lượng"]).sum() / tong_so)
    tb_chi_tieu_chung = float((cum_hop_le["Chi tiêu TB"] * cum_hop_le["Số lượng"]).sum() / tong_so)

    dem_ten: Dict[str, int] = {}
    ket_qua_so_bo = []
    for _, row in profile_df.iterrows():
        cum_id = int(row["Cluster"])
        if cum_id == -1:
            ten_persona = "Khách hàng Ngoại lai / Hành vi đặc biệt"
            chien_luoc = "Kiểm tra thủ công lịch sử giao dịch để loại trừ dữ liệu bất thường hoặc phục vụ riêng."
        else:
            muc_thu_nhap = _phan_muc(float(row["Thu nhập TB"]), tb_thu_nhap_chung)
            muc_chi_tieu = _phan_muc(float(row["Chi tiêu TB"]), tb_chi_tieu_chung)
            ten_co_ban, chien_luoc = _CHIEN_LUOC_THEO_NHOM.get(
                (muc_thu_nhap, muc_chi_tieu),
                (
                    "Khách hàng Tiêu chuẩn / Trung bình",
                    "Duy trì tần suất mua sắm đều đặn qua email marketing, voucher sinh nhật và tích điểm thành viên.",
                ),
            )
            nhan_do_tuoi = "Trẻ/Trung niên trẻ" if float(row["Tuổi TB"]) < tb_tuoi_chung - 2.0 else (
                "Trung niên/Cao tuổi" if float(row["Tuổi TB"]) > tb_tuoi_chung + 2.0 else "Độ tuổi phổ thông"
            )
            ten_persona = f"{ten_co_ban} ({nhan_do_tuoi})"

        dem_ten[ten_persona] = dem_ten.get(ten_persona, 0) + 1
        ket_qua_so_bo.append((cum_id, row, ten_persona, chien_luoc))

    # Gắn chỉ số phân biệt nếu có từ 2 cụm trùng tên Persona
    stt_ten: Dict[str, int] = {}
    personas: List[Dict[str, Any]] = []
    for cum_id, row, ten_persona, chien_luoc in ket_qua_so_bo:
        ten_cuoi = ten_persona
        if dem_ten[ten_persona] > 1:
            stt_ten[ten_persona] = stt_ten.get(ten_persona, 0) + 1
            ten_cuoi = f"{ten_persona} #{stt_ten[ten_persona]}"

        personas.append(
            {
                "cluster_id": cum_id,
                "persona_name": ten_cuoi,
                "size": int(row["Số lượng"]),
                "pct": float(row["Tỷ lệ (%)"]),
                "avg_age": float(row["Tuổi TB"]),
                "avg_income": float(row["Thu nhập TB"]),
                "avg_spending": float(row["Chi tiêu TB"]),
                "marketing_strategy": chien_luoc,
            }
        )

    print(f" [ĐỊNH DANH PERSONA] Đã xây dựng chân dung và chiến lược cho {len(personas)} phân khúc.")
    return personas

"""
=============================================================================
MODULE: PHÂN TÍCH HỒ SƠ PHÂN KHÚC KHÁCH HÀNG (CUSTOMER PROFILING & PERSONAS)
=============================================================================
Mục đích:
- Phân tích thống kê đặc trưng nhân khẩu học và hành vi tiêu dùng theo từng cụm.
- Định danh Persona kinh doanh cho từng phân khúc khách hàng.
- Đề xuất chiến lược tiếp thị (Marketing Strategies) và chăm sóc khách hàng (CRM)
  hành động thực tế mang lại giá trị gia tăng cho doanh nghiệp bán lẻ.
=============================================================================
"""

from typing import Dict, Any, List
import pandas as pd
from src.config import AGE_COLUMN, INCOME_COLUMN, SPENDING_COLUMN, METRICS_DIR


def create_cluster_profiles(df: pd.DataFrame, labels: pd.Series | list) -> pd.DataFrame:
    """
    Ý LÀM:
    1. Gắn nhãn phân cụm (Cluster) vào DataFrame gốc của khách hàng.
    2. Nhóm theo từng cụm (groupby('Cluster')) và tính các thống kê mô tả:
       - Số lượng khách hàng và tỷ lệ phần trăm trong tổng số.
       - Tuổi trung bình và trung vị.
       - Thu nhập trung bình và trung vị.
       - Điểm chi tiêu trung bình và trung vị.
       - Tỷ lệ giới tính Nam / Nữ trong từng cụm.
    3. Lưu bảng kết quả vào results/metrics/cluster_profiles.csv.
    4. Trả về DataFrame hồ sơ chi tiết từng cụm.
    """
    pass


def generate_customer_personas(profile_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Ý LÀM (Xây dựng Chân dung khách hàng - Personas & Đề xuất Marketing):
    Dựa vào tọa độ trung tâm của 5 cụm kinh điển, gán nhãn Persona tương ứng:
    - Nhóm 1: "Khách hàng Tiêu chuẩn / Phổ thông" (Thu nhập TB, Chi tiêu TB)
      -> Chiến lược: Duy trì tần suất mua sắm bằng voucher giảm giá định kỳ, email marketing.
    - Nhóm 2: "Khách hàng VIP / Ngôi sao" (Thu nhập Cao, Chi tiêu Cao)
      -> Chiến lược: Dịch vụ khách hàng cao cấp, đặc quyền thành viên VIP, sự kiện ra mắt độc quyền.
    - Nhóm 3: "Khách hàng Cẩn trọng / Tiết kiệm cao cấp" (Thu nhập Cao, Chi tiêu Thấp)
      -> Chiến lược: Quảng bá sản phẩm chất lượng cao, có giá trị đầu tư lâu dài, chính sách bảo hành vàng.
    - Nhóm 4: "Khách hàng Trẻ / Chi tiêu bốc đồng" (Thu nhập Thấp, Chi tiêu Cao)
      -> Chiến lược: Chiến dịch khuyến mãi Flash sale, sản phẩm bắt trend (xu hướng), trả góp 0%.
    - Nhóm 5: "Khách hàng Tiết kiệm / Bình dân" (Thu nhập Thấp, Chi tiêu Thấp)
      -> Chiến lược: Combo giá rẻ, chương trình tích điểm đổi quà, giảm giá xả kho.
    """
    pass

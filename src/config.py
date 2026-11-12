"""
=============================================================================
MODULE: CẤU HÌNH HỆ THỐNG DỰ ÁN (CONFIG)
=============================================================================
Mục đích:
- Quản lý tập trung toàn bộ cấu hình, hằng số, đường dẫn và siêu tham số của dự án.
- Thiết lập đường dẫn thư mục chuẩn: data/raw, data/processed, results/figures,
  results/metrics, results/saved_models.
- Cố định Random Seed để đảm bảo tính tái lập (reproducibility) trong toàn bộ pipeline.
- Định nghĩa tên các thuộc tính của tập dữ liệu khách hàng (Marketing Campaign Dataset).
- Cung cấp cấu hình thẩm mỹ hiển thị (Seaborn, Matplotlib, Bảng màu doanh nghiệp, Metric Cards).
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Không tắt warnings toàn cục để đảm bảo cảnh báo dữ liệu/mô hình luôn được ghi nhận minh bạch

# =============================================================================
# 1. ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN THƯ MỤC CHUẨN TRONG DỰ ÁN
# =============================================================================
# Thư mục gốc của project (lab_3.1)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILENAME = "marketing_campaign.csv"
DEFAULT_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
RAW_DATA_PATH = DEFAULT_DATA_PATH
DATA_SEPARATOR = "\t"

# Thư mục kết quả đầu ra (Results)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"           # Lưu trữ các biểu đồ trực quan hóa (.png)
METRICS_DIR = RESULTS_DIR / "metrics"           # Lưu trữ các file số liệu đánh giá (.csv, .json)
SAVED_MODELS_DIR = RESULTS_DIR / "saved_models" # Lưu trữ mô hình và preprocessor (.pkl)

# Thư mục tài liệu và sổ tay
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Tự động khởi tạo cấu trúc thư mục nếu chưa tồn tại trên ổ đĩa
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    SAVED_MODELS_DIR,
    NOTEBOOKS_DIR,
    TESTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. CẤU HÌNH TÁI LẬP (REPRODUCIBILITY) & HẰNG SỐ MÔ HÌNH
# =============================================================================
# Cố định hạt giống ngẫu nhiên để các thuật toán K-Means, chia tập, PCA luôn ra kết quả giống nhau
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Tỷ lệ phân chia tập dữ liệu chuẩn mực (80% Train, 20% Holdout Test)
TRAIN_TEST_SPLIT_RATIO = 0.20

# Phạm vi số cụm K cần quét trong thuật toán Elbow Method và Silhouette Analysis
K_SEARCH_RANGE = range(2, 11)

# =============================================================================
# 3. TÊN CỘT THUỘC TÍNH DỮ LIỆU KHÁCH HÀNG (CUSTOMER DATASET SCHEMA)
# =============================================================================
# Thuộc tính định danh (không tham gia gom cụm)
ID_COLUMN = "ID"

# Mốc thời gian chuẩn để tính tuổi khách hàng (Dt_Customer tối đa trong tập là năm 2014)
REFERENCE_YEAR = 2014
MIN_VALID_YEAR_BIRTH = 1920
MAX_VALID_INCOME = 200000.0

# Cột thô cần thiết từ marketing_campaign.csv
YEAR_BIRTH_COLUMN = "Year_Birth"
SPENDING_COMPONENT_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
REQUIRED_RAW_COLUMNS = [
    ID_COLUMN,
    YEAR_BIRTH_COLUMN,
    "Income",
    *SPENDING_COMPONENT_COLUMNS,
]

# Các cột phân loại nhân khẩu học phục vụ EDA / Profiling (không có Gender trong tập gốc)
CATEGORICAL_PROFILE_COLUMNS = ["Education", "Marital_Status"]

# Các đặc trưng số dùng để phân cụm sau khi biến đổi từ dữ liệu gốc
AGE_COLUMN = "Age"                   # Độ tuổi khách hàng = REFERENCE_YEAR - Year_Birth
INCOME_COLUMN = "Income"             # Thu nhập hộ gia đình hàng năm (USD)
SPENDING_COLUMN = "Total_Spending"   # Tổng chi tiêu trên 6 nhóm sản phẩm Mnt*

# Danh sách đầy đủ các đặc trưng số dùng để phân cụm
NUMERICAL_FEATURES = [AGE_COLUMN, INCOME_COLUMN, SPENDING_COLUMN]

# Cặp đặc trưng 2D cốt lõi thể hiện rõ nhất phân khúc thu nhập - chi tiêu
KEY_2D_FEATURES = [INCOME_COLUMN, SPENDING_COLUMN]

# =============================================================================
# 4. CẤU HÌNH GIAO DIỆN & BẢNG MÀU CHUYÊN NGHIỆP (STYLING & PALETTES)
# =============================================================================
# Bảng mã màu chuẩn doanh nghiệp hiện đại
STYLE = {
    "primary": "#1E3A8A",       # Xanh dương hoàng gia (Royal Blue)
    "primary_dark": "#0F172A",  # Đen xám đậm (Slate Dark)
    "secondary": "#0D9488",     # Xanh ngọc lục bảo (Teal)
    "blue": "#2563EB",          # Xanh dương tươi sáng
    "soft_blue": "#EFF6FF",     # Xanh nhạt nền
    "green": "#16A34A",         # Xanh lá cây (Thành công / Cụm tích cực)
    "soft_green": "#F0FDF4",    # Xanh lá pastel
    "yellow": "#D97706",        # Vàng hổ phách (Cảnh báo / Cụm trung bình)
    "soft_yellow": "#FFFBEB",   # Vàng nhạt nền
    "red": "#DC2626",           # Đỏ tươi (Ngoại lai / Nguy cơ)
    "soft_red": "#FEF2F2",      # Đỏ pastel
    "purple": "#7C3AED",        # Tím đậm
    "soft_purple": "#F5F3FF",   # Tím pastel
    "gray": "#F8FAFC",          # Xám nhạt nền
    "border": "#E2E8F0",        # Màu viền thanh lịch
    "text": "#1E293B",          # Màu chữ than chì dễ đọc
}

# Bảng màu đại diện cho các cụm khách hàng phân biệt rõ ràng
CLUSTER_PALETTE = [
    "#2563EB",  # Cụm 0: Xanh dương
    "#EA580C",  # Cụm 1: Cam đậm
    "#16A34A",  # Cụm 2: Xanh lá cây
    "#DC2626",  # Cụm 3: Đỏ tươi
    "#9333EA",  # Cụm 4: Tím thẫm
    "#0D9488",  # Cụm 5: Xanh ngọc
    "#CA8A04",  # Cụm 6: Vàng đồng
    "#DB2777",  # Cụm 7: Hồng sen
]

# Cấu hình thông số mặc định cho đồ họa Matplotlib
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")


# =============================================================================
# 5. CÁC HÀM TIỆN ÍCH TRÌNH BÀY & XUẤT ĐỒ THỊ (DISPLAY & EXPORT HELPERS)
# =============================================================================
def save_figure(filename: str, dpi: int = 200) -> Path:
    """
    Lưu đồ thị hiện tại vào thư mục results/figures/ với độ phân giải cao (DPI 200).

    Tham số:
        filename: Tên file ảnh kèm đuôi (.png).
        dpi: Độ sắc nét của ảnh (mặc định 200).

    Trả về:
        Đường dẫn Path tới file ảnh vừa lưu.
    """
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f" [ĐÃ LƯU BIỂU ĐỒ] -> {path.relative_to(PROJECT_ROOT)}")
    return path


def print_section(title: str, subtitle: str | None = None):
    """
    In tiêu đề phân mục đẹp mắt, rõ ràng trên Terminal Console.
    """
    separator = "=" * 78
    print(f"\n{separator}")
    print(f"  >>> {title.upper()}")
    if subtitle:
        print(f"      {subtitle}")
    print(f"{separator}\n")

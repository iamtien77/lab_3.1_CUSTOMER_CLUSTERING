"""
=============================================================================
MODULE: LƯU TRỮ VÀ TẢI MÔ HÌNH (MODEL PERSISTENCE)
=============================================================================
Mục đích:
- Đóng gói toàn bộ mô hình (Preprocessor Scaler + Mô hình Phân Cụm + Metadata)
  bằng thư viện joblib vào thư mục results/saved_models/.
- Hỗ trợ nạp lại mô hình đã huấn luyện để phục vụ dự đoán và suy luận thời gian thực.
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional
import joblib
from src.config import SAVED_MODELS_DIR


def save_model(
    model_object: Any,
    scaler_object: Any,
    model_name: str = "final_kmeans_pipeline.pkl",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Ý LÀM:
    1. Đóng gói một Dictionary hoàn chỉnh gồm:
       - 'model': Mô hình phân cụm đã fit (KMeans, DBSCAN, ...).
       - 'scaler': Bộ chuẩn hóa dữ liệu đã fit (FeatureScaler).
       - 'metadata': Thông tin cấu hình (tên đặc trưng, số cụm, ngày tạo, chỉ số đánh giá).
    2. Xác định đường dẫn file lưu: results/saved_models/{model_name}.
    3. Dùng joblib.dump() để ghi đối tượng xuống đĩa.
    4. Trả về đường dẫn Path tới file đã lưu.
    """
    pass


def load_model(model_name: str = "final_kmeans_pipeline.pkl") -> Dict[str, Any]:
    """
    Ý LÀM:
    1. Kiểm tra sự tồn tại của file mô hình trong thư mục results/saved_models/.
    2. Dùng joblib.load() để giải nén và nạp gói mô hình vào bộ nhớ.
    3. Trả về dictionary chứa 'model', 'scaler', 'metadata' sẵn sàng cho việc suy luận.
    """
    pass

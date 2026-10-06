"""
=============================================================================
MODULE: TRIỂN KHAI VÀ DỰ ĐOÁN PHÂN KHÚC KHÁCH HÀNG MỚI (INFERENCE / PREDICTOR)
=============================================================================
Mục đích:
- Nạp gói mô hình hoàn chỉnh đã lưu trữ từ thư mục results/saved_models/.
- Cung cấp API suy luận thời gian thực cho khách hàng mới (chưa từng xuất hiện trong tập train).
- Tự động chuẩn hóa dữ liệu đầu vào bằng bộ Scaler đã học.
- Dự đoán cụm (Cluster ID), gán tên Persona kinh doanh và tính khoảng cách tới tâm cụm
  để đánh giá mức độ tin cậy của việc gán nhóm.
=============================================================================
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.models.persistence import load_model


class CustomerSegmentPredictor:
    """
    Lớp phục vụ dự đoán phân khúc cho khách hàng mới.
    """

    def __init__(self, model_filename: str = "final_kmeans_pipeline.pkl"):
        """
        Ý LÀM:
        1. Gọi hàm load_model(model_filename) để nạp dictionary chứa:
           - self.model: Mô hình phân cụm đã huấn luyện.
           - self.scaler: Bộ chuẩn hóa đặc trưng tương ứng.
           - self.metadata: Tên đặc trưng, số cụm, định nghĩa personas.
        2. Lưu trữ các thuộc tính sẵn sàng cho việc nhận request dự đoán.
        """
        pass

    def predict_single_customer(
        self,
        age: int,
        annual_income: float,
        spending_score: float,
        gender: Optional[str] = "Female",
    ) -> Dict[str, Any]:
        """
        Ý LÀM:
        1. Đóng gói dữ liệu của 1 khách hàng mới thành DataFrame 1 dòng.
        2. Dùng self.scaler.transform() để chuẩn hóa các đặc trưng theo đúng thang đo đã học.
        3. Dùng self.model.predict() để gán nhãn cụm (cluster_id).
        4. Tính khoảng cách Euclidean từ điểm này tới tâm của cụm được gán (Centroid distance).
        5. Ánh xạ cluster_id sang tên Persona kinh doanh và chiến lược tiếp thị phù hợp.
        6. Trả về kết quả dự đoán dạng Dictionary:
           {
               'customer_input': {'age': age, 'income': annual_income, 'spending': spending_score},
               'assigned_cluster': cluster_id,
               'persona_name': 'Khách hàng VIP / Ngôi sao',
               'marketing_recommendation': 'Tặng voucher VIP và dịch vụ chăm sóc cao cấp',
               'distance_to_center': 0.42
           }
        """
        pass

    def predict_batch(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Ý LÀM:
        1. Nhận vào bảng DataFrame chứa danh sách nhiều khách hàng mới.
        2. Chuẩn hóa đặc trưng hàng loạt bằng scaler.transform().
        3. Dự đoán nhãn cụm hàng loạt bằng model.predict().
        4. Gắn thêm 2 cột mới vào DataFrame: ['Predicted_Cluster', 'Persona_Name'].
        5. Trả về DataFrame kết quả đã được gán phân khúc.
        """
        pass

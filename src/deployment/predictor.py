"""
=============================================================================
MODULE: TRIỂN KHAI VÀ DỰ ĐOÁN PHÂN KHÚC KHÁCH HÀNG MỚI (INFERENCE / PREDICTOR)
=============================================================================
Mục đích (mức độ bài lab học tập - CLI/Python):
- Nạp gói mô hình hoàn chỉnh đã lưu từ thư mục results/saved_models/.
- Tự động chuẩn hóa dữ liệu đầu vào bằng bộ Scaler đã fit trên tập Train.
- Gán nhãn cụm, xác định tên Persona và tính khoảng cách tới tâm cụm đại diện.
=============================================================================
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.config import AGE_COLUMN, INCOME_COLUMN, SPENDING_COLUMN
from src.data.preprocessing import clean_data
from src.models.persistence import load_model


class CustomerSegmentPredictor:
    """
    Lớp suy luận phân khúc cho khách hàng mới dựa trên pipeline đã huấn luyện.
    """

    def __init__(
        self,
        model_filename: str = "final_kmeans_pipeline.pkl",
        model_dir: Optional[Path | str] = None,
    ):
        """Nạp gói mô hình gồm model, scaler và metadata chứa định nghĩa Persona."""
        model_bundle = load_model(model_name=model_filename, input_dir=model_dir)
        self.model = model_bundle["model"]
        if not hasattr(self.model, "predict"):
            raise TypeError("Đối tượng mô hình nạp vào không hỗ trợ phương thức predict().")
        self.scaler = model_bundle["scaler"]
        self.metadata = model_bundle.get("metadata", {})
        self._personas_map = self._normalize_persona_table(self.metadata.get("personas", {}))

    @staticmethod
    def _normalize_persona_table(raw_personas: Any) -> Dict[int, Dict[str, Any]]:
        """Chuyển đổi danh sách hoặc từ điển Persona trong metadata sang map theo cluster_id."""
        if isinstance(raw_personas, dict):
            return {int(k): v for k, v in raw_personas.items()}
        if isinstance(raw_personas, list):
            return {int(p["cluster_id"]): p for p in raw_personas if "cluster_id" in p}
        return {}

    def _lookup_persona(self, cluster_id: int) -> tuple[str, str]:
        """Lấy tên Persona và khuyến nghị tiếp thị tương ứng với mã cụm."""
        persona_info = self._personas_map.get(int(cluster_id))
        if not persona_info:
            return f"Cụm {cluster_id}", "Chưa có mô tả Persona riêng trong metadata."
        name = persona_info.get("persona_name", f"Cụm {cluster_id}")
        strategy = persona_info.get("marketing_strategy", "")
        return name, strategy

    def predict_single_customer(
        self,
        age: float,
        annual_income: float,
        total_spending: float,
    ) -> Dict[str, Any]:
        """
        Dự đoán phân khúc và khoảng cách tới tâm cụm cho 1 khách hàng mới.
        """
        if not np.all(np.isfinite([age, annual_income, total_spending])):
            raise ValueError("Age, Income và Total_Spending phải là các giá trị số hữu hạn (không chứa NaN/Inf).")
        if age <= 0 or age > 120:
            raise ValueError(f"Độ tuổi không hợp lệ: {age}. Phải nằm trong khoảng (0, 120].")
        if annual_income < 0 or total_spending < 0:
            raise ValueError("Thu nhập và tổng chi tiêu không được âm.")

        single_row_df = pd.DataFrame(
            [
                {
                    AGE_COLUMN: float(age),
                    INCOME_COLUMN: float(annual_income),
                    SPENDING_COLUMN: float(total_spending),
                }
            ]
        )
        scaled_matrix = self.scaler.transform(single_row_df)
        cluster_id = int(self.model.predict(scaled_matrix)[0])

        distance = float("nan")
        if hasattr(self.model, "cluster_centers_"):
            cluster_center = self.model.cluster_centers_[cluster_id]
            delta = scaled_matrix[0] - cluster_center
            if getattr(self.model, "_scipy_metric", None) == "cityblock":
                distance = round(float(np.sum(np.abs(delta))), 4)
            else:
                distance = round(float(np.linalg.norm(delta)), 4)

        persona_name, recommendation = self._lookup_persona(cluster_id)
        return {
            "customer_input": {
                AGE_COLUMN: float(age),
                INCOME_COLUMN: float(annual_income),
                SPENDING_COLUMN: float(total_spending),
            },
            "assigned_cluster": cluster_id,
            "persona_name": persona_name,
            "marketing_recommendation": recommendation,
            "distance_to_center": distance,
        }

    def predict_batch(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Dự đoán phân khúc hàng loạt cho bảng khách hàng mới (tự tính Age/Total_Spending và điền thiếu an toàn).
        """
        if df_new.empty:
            raise ValueError("DataFrame khách hàng mới đang rỗng, không thể dự đoán.")

        # Dùng trung vị đã lưu từ tập Train (nếu có) để điền thiếu mà không gây Data Leakage
        train_medians = self.metadata.get("feature_medians")
        result_df = clean_data(
            df_new,
            fill_medians=train_medians,
            drop_invalid_outliers=False,
            derive_features=True,
            preserve_index=True,
        )
        scaled_matrix = self.scaler.transform(result_df)
        predicted_labels = self.model.predict(scaled_matrix)

        persona_names = []
        recommendations = []
        for cluster_id in predicted_labels:
            name, strategy = self._lookup_persona(int(cluster_id))
            persona_names.append(name)
            recommendations.append(strategy)

        result_df["Predicted_Cluster"] = predicted_labels.astype(int)
        result_df["Persona_Name"] = persona_names
        result_df["Marketing_Recommendation"] = recommendations
        print(f" [SUY LUẬN HÀNG LOẠT] Đã gắn nhãn phân khúc cho {len(result_df):,} khách hàng mới.")
        return result_df


def run_cli():
    """Giao diện dòng lệnh (CLI) mức bài lab để dự đoán phân khúc 1 khách hàng mới."""
    parser = argparse.ArgumentParser(description="Dự đoán phân khúc cho 1 khách hàng mới từ mô hình đã lưu.")
    parser.add_argument("--age", type=float, required=True, help="Độ tuổi khách hàng (năm)")
    parser.add_argument("--income", type=float, required=True, help="Thu nhập hộ gia đình hàng năm (USD)")
    parser.add_argument("--spending", type=float, required=True, help="Tổng chi tiêu (USD)")
    parser.add_argument("--model", type=str, default="final_kmeans_pipeline.pkl", help="Tên tệp mô hình trong results/saved_models/")
    parser.add_argument("--model-dir", type=str, default=None, help="Thư mục tùy chọn chứa tệp mô hình (.pkl)")
    args = parser.parse_args()

    predictor = CustomerSegmentPredictor(model_filename=args.model, model_dir=args.model_dir)
    prediction = predictor.predict_single_customer(
        age=args.age,
        annual_income=args.income,
        total_spending=args.spending,
    )
    print(f"Cụm dự đoán: {prediction['assigned_cluster']} - {prediction['persona_name']}")
    print(f"Khoảng cách tới tâm cụm: {prediction['distance_to_center']}")
    print(f"Chiến lược gợi ý: {prediction['marketing_recommendation']}")


if __name__ == "__main__":
    run_cli()

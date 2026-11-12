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

from src.config import (
    AGE_COLUMN, INCOME_COLUMN, SPENDING_COLUMN, NUMERICAL_FEATURES,
    REFERENCE_YEAR, SPENDING_COMPONENT_COLUMNS, YEAR_BIRTH_COLUMN,
)
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

    @staticmethod
    def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
        """Chỉ cho phép số hữu hạn hoặc giá trị khuyết thật; báo index của dòng lỗi."""
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = (df[column].notna() & values.isna()) | (
            values.notna() & ~np.isfinite(values)
        )
        if invalid.any():
            raise ValueError(f"Cột '{column}' không phải số hữu hạn tại index {df.index[invalid].tolist()}.")
        return values

    @classmethod
    def _validate_features(cls, df: pd.DataFrame, allow_missing: bool = False) -> None:
        """Dùng chung miền dữ liệu cho single/batch; không loại bỏ dòng âm thầm."""
        for column in NUMERICAL_FEATURES:
            if column not in df:
                raise ValueError(f"Thiếu cột đặc trưng '{column}'.")
            values = cls._numeric_column(df, column)
            invalid = ((values <= 0) | (values > 120)) if column == AGE_COLUMN else values < 0
            if not allow_missing:
                invalid = invalid | values.isna()
            if invalid.any():
                raise ValueError(f"Cột '{column}' sai miền dữ liệu tại index {df.index[invalid].tolist()}.")

    def predict_single_customer(
        self,
        age: float,
        annual_income: float,
        total_spending: float,
    ) -> Dict[str, Any]:
        """
        Dự đoán phân khúc và khoảng cách tới tâm cụm cho 1 khách hàng mới.
        """
        single_row_df = pd.DataFrame(
            [
                {
                    AGE_COLUMN: float(age),
                    INCOME_COLUMN: float(annual_income),
                    SPENDING_COLUMN: float(total_spending),
                }
            ]
        )
        self._validate_features(single_row_df)
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

        prepared_df = df_new.copy()
        if YEAR_BIRTH_COLUMN in prepared_df:
            prepared_df[AGE_COLUMN] = self.metadata.get("reference_year", REFERENCE_YEAR) - self._numeric_column(
                prepared_df, YEAR_BIRTH_COLUMN
            )
        for column in SPENDING_COMPONENT_COLUMNS:
            if column in prepared_df:
                values = self._numeric_column(prepared_df, column)
                if (values < 0).any():
                    raise ValueError(f"Cột '{column}' âm tại index {prepared_df.index[values < 0].tolist()}.")
                prepared_df[column] = values
        if all(column in prepared_df for column in SPENDING_COMPONENT_COLUMNS):
            # Khoản chi khuyết không đồng nghĩa với không chi tiêu.
            prepared_df[SPENDING_COLUMN] = prepared_df[SPENDING_COMPONENT_COLUMNS].sum(
                axis=1, min_count=len(SPENDING_COMPONENT_COLUMNS)
            )
        self._validate_features(prepared_df, allow_missing=True)
        train_medians = self.metadata.get("feature_medians")
        for column in NUMERICAL_FEATURES:
            prepared_df[column] = self._numeric_column(prepared_df, column)
            if prepared_df[column].isna().any():
                try:
                    median = float(train_medians[column])
                except (TypeError, KeyError, ValueError):
                    raise ValueError(f"Thiếu trung vị Train hợp lệ cho cột '{column}' trong feature_medians.") from None
                if not np.isfinite(median):
                    raise ValueError(f"Trung vị Train của '{column}' phải hữu hạn.")
                prepared_df[column] = prepared_df[column].fillna(median)
        self._validate_features(prepared_df)
        result_df = clean_data(
            prepared_df,
            fill_medians=train_medians,
            drop_invalid_outliers=False,
            derive_features=False,
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

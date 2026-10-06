"""
Package đánh giá chất lượng phân cụm và trực quan hóa (Evaluation Package).
"""

from src.evaluation.metrics import evaluate_clustering, compare_all_models
from src.evaluation.visualization import (
    plot_elbow_and_silhouette,
    plot_clusters_2d,
    plot_clusters_3d,
    plot_dendrogram,
    plot_silhouette_analysis,
)
from src.evaluation.profiling import create_cluster_profiles, generate_customer_personas

__all__ = [
    "evaluate_clustering",
    "compare_all_models",
    "plot_elbow_and_silhouette",
    "plot_clusters_2d",
    "plot_clusters_3d",
    "plot_dendrogram",
    "plot_silhouette_analysis",
    "create_cluster_profiles",
    "generate_customer_personas",
]

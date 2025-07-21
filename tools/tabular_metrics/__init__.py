from typing import Dict

from .classification import ClassificationBenchmark
from .regression import RegressionBenchmark


def evaluate_classification(X, y, model, use_tensor=False) -> Dict[str, float]:
    bench = ClassificationBenchmark(X, y, model, use_tensor=use_tensor)

    return {
        "accuracy": float(bench.accuracy()),
        "precision": float(bench.precision()),
        "recall": float(bench.recall()),
        "f1": float(bench.F1()),
        "roc_auc": float(bench.roc_auc()),
        "informedness": float(bench.informedness()),
        "markedness": float(bench.markedness()),
        "matthews_corrcoef": float(bench.matthews()),
    }


def evaluate_regression(X, y, model, use_tensor=False) -> Dict[str, float]:
    bench = RegressionBenchmark(X, y, model, use_tensor=use_tensor)

    return {
        "mae": float(bench.mae()),
        "nmae": float(bench.nmae()),
        "rmse": float(bench.rmse()),
        "nrmse": float(bench.nrmse()),
        "r2": float(bench.r2()),
        "adj_r2": float(bench.adj_r2()),
    }

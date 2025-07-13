from typing import Dict

from .classification import ClassificationBenchmark
from .regression import RegressionBenchmark


def evaluate_classification(X, y, model) -> Dict[str, float]:
    bench = ClassificationBenchmark(X, y, model)
    
    return {
        'accuracy': bench.accuracy(),
        'precision': bench.precision(),
        'recall': bench.recall(),
        'F1': bench.F1(),
        'roc_auc': bench.roc_auc(),
        'informedness': bench.informedness(),
        'markedness': bench.markedness(),
        'matthews_corrcoef': bench.matthews(),
        'model_name': bench.model_name,
        'dim': bench.dim,
        'n_samples': bench.n
    }

def evaluate_regression(X, y, model) -> Dict[str, float]:
    bench = RegressionBenchmark(X, y, model)
    
    return {
        'mae': bench.mae(),
        'nmae': bench.nmae(),
        'rmse': bench.rmse(),
        'nrmse': bench.nrmse(),
        'r2': bench.r2(),
        'adj_r2': bench.adj_r2(),
        'model_name': bench.model_name,
        'dim': bench.dim,
        'n_samples': bench.n
    }

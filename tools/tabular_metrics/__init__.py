from typing import Dict

from .classification import *
from .regression import *


def get_classification_metrics(X, y, model) -> Dict[str, float]:
    raise NotImplementedError(
        "get_classification_metrics is not implemented. Please implement it in the classification module."
    )


def get_regression_metrics(X, y, model) -> Dict[str, float]:
    raise NotImplementedError(
        "get_regression_metrics is not implemented. Please implement it in the regression module."
    )

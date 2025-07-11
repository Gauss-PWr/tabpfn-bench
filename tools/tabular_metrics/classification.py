from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)

"""
===============================
Classification
===============================
"""


def automl_benchmark_metric(target, pred, numpy=False, should_raise=False):
    lib = np if numpy else torch

    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    if len(lib.unique(target)) > 2:
        return -cross_entropy(target, pred)
    else:
        return auc_metric_ovr(target, pred, numpy=numpy, should_raise=should_raise)


def auc_metric_ovr(target, pred, numpy=False, should_raise=False):
    return auc_metric(
        target, pred, multi_class="ovr", numpy=numpy, should_raise=should_raise
    )


def auc_metric_ovo(target, pred, numpy=False, should_raise=False):
    return auc_metric(
        target, pred, multi_class="ovo", numpy=numpy, should_raise=should_raise
    )


def remove_classes_not_in_target_from_pred(target, pred):
    assert torch.is_tensor(target) == torch.is_tensor(
        pred
    ), "target and pred must be both torch tensors or both numpy arrays"
    convert_to_torch = False
    if torch.is_tensor(target):
        convert_to_torch = True
        target = target.numpy()
        pred = pred.numpy()
    pred = pred.copy()
    target = target.copy()

    unique_targets = np.unique(target)
    assert all(
        unique_targets[:-1] <= unique_targets[1:]
    ), "target must be sorted after unique"

    # assumption is that target is 0-indexed before removing classes
    if len(unique_targets) < pred.shape[1]:
        assert (
            unique_targets < pred.shape[1]
        ).all(), "target must be smaller than pred.shape[1]"
        pred = pred[:, unique_targets]
        pred = pred / pred.sum(axis=1, keepdims=True)

        if np.isnan(np.sum(pred)):
            # Nan values as a result of adjustment, make it very small probability and equalize again.
            # Nan can happen if pred.sum() above is 0 due to not having given any likelihood to all but the removed classes.
            nan_mask = np.isnan(pred).any(axis=1)
            e = np.finfo(float).eps
            pred[nan_mask] = np.nan_to_num(pred[nan_mask], nan=e, posinf=e, neginf=e)
            pred[nan_mask] = pred[nan_mask] / pred[nan_mask].sum(axis=1, keepdims=True)

        # make target 0-indexed again, just for beauty
        # sklearn would handle it anyway
        mapping = {c: i for i, c in enumerate(unique_targets)}
        target = np.array([mapping[c] for c in target])
    if convert_to_torch:
        target = torch.tensor(target)
        pred = torch.tensor(pred)
    return target, pred


def auc_metric(target, pred, multi_class="ovr", numpy=False, should_raise=False):
    lib = np if numpy else torch

    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    else:
        target = np.array(target)
        pred = np.array(pred)

    # When using sklearn's cross val score with this function, it expects the metric to accept 1D prediction in the binary case
    # Hence, we only apply our fix here, if pred shape is 2D (a.k.a. multiclass or called not from sklearn)
    if len(pred.shape) > 1:
        target, pred = remove_classes_not_in_target_from_pred(target, pred)
        assert (
            len(lib.unique(target)) == pred.shape[1]
        ), "target and pred must have the same number of classes"

        if pred.shape[1] == 2:
            pred = pred[:, 1]

    score = roc_auc_score(target, pred, multi_class=multi_class)
    if not numpy:
        return torch.tensor(score)
    return score


def accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(accuracy_score(target, pred[:, 1] > 0.5))


def f1_metric(target, pred, multi_class="micro"):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(
            f1_score(target, torch.argmax(pred, -1), average=multi_class)
        )
    else:
        return torch.tensor(f1_score(target, pred[:, 1] > 0.5))


def average_precision_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(average_precision_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(average_precision_score(target, pred[:, 1] > 0.5))


def balanced_accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(balanced_accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(balanced_accuracy_score(target, pred[:, 1] > 0.5))


def cross_entropy(target, pred, numpy=False):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        ce = torch.nn.CrossEntropyLoss()
        return ce(pred.float().log(), target.long())
    else:
        bce = torch.nn.BCELoss()
        return bce(pred[:, 1].float(), target.float())


def is_classification(metric_used):
    if metric_used == auc_metric or metric_used == cross_entropy:
        return True
    return False


def nll_bar_dist(target, pred, bar_dist):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    target, pred = target.unsqueeze(0).to(bar_dist.borders.device), pred.unsqueeze(
        1
    ).to(bar_dist.borders.device)

    l = bar_dist(pred.log(), target).mean().cpu()
    return l


def expected_calibration_error(target, pred, norm="l1", n_bins=10):
    import torchmetrics

    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    target, pred = remove_classes_not_in_target_from_pred(target, pred)

    ece = torchmetrics.classification.MulticlassCalibrationError(
        n_bins=n_bins,
        norm=norm,
        num_classes=len(torch.unique(target)),
    )
    return ece(
        target=target,
        preds=pred,
    )


def is_imbalanced(y, threshold=0.8):
    """
    Determine if a numpy array of class labels is imbalanced based on Gini impurity.

    Parameters:
    - y (numpy.ndarray): A 1D numpy array containing class labels.
    - threshold (float): Proportion of the maximum Gini impurity to consider as the boundary
                         between balanced and imbalanced. Defaults to 0.8.

    Returns:
    - bool: True if the dataset is imbalanced, False otherwise.

    Example:
    >>> y = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    >>> is_imbalanced(y)
    True
    """

    # Calculate class proportions
    _, class_counts = np.unique(y, return_counts=True)
    class_probs = class_counts / len(y)

    # Calculate Gini impurity
    gini = 1 - np.sum(class_probs**2)

    # Determine max possible Gini for the number of classes
    C = len(class_probs)
    max_gini = 1 - 1 / C

    # Check if the Gini impurity is less than the threshold of the maximum possible Gini
    return gini < threshold * max_gini

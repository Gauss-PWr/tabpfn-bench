from __future__ import annotations

import torch
from sklearn.metrics import mean_absolute_error, r2_score

"""
===============================
Regression
===============================
"""


def root_mean_squared_error_metric(target, pred, normalize=False):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    target_ = (
        target
        if not normalize
        else (target - target.min()) / (target.max() - target.min())
    )
    pred = (
        pred if not normalize else (pred - target.min()) / (target.max() - target.min())
    )

    return torch.sqrt(torch.nn.functional.mse_loss(target_, pred))


def normalized_root_mean_squared_error_metric(target, pred):
    return root_mean_squared_error_metric(target, pred, normalize=True)


def mean_squared_error_metric(target, pred, normalize=False):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    target_ = (
        target
        if not normalize
        else (target - target.min()) / (target.max() - target.min())
    )
    pred = (
        pred if not normalize else (pred - target.min()) / (target.max() - target.min())
    )

    return torch.nn.functional.mse_loss(target_, pred)


def normalized_mean_squared_error_metric(target, pred):
    return mean_squared_error_metric(target, pred, normalize=True)


def mean_absolute_error_metric(target, pred, normalize=False):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    target_ = (
        target
        if not normalize
        else (target - target.min()) / (target.max() - target.min())
    )
    pred = (
        pred if not normalize else (pred - target.min()) / (target.max() - target.min())
    )

    return torch.tensor(mean_absolute_error(target_, pred))


def normalized_mean_absolute_error_metric(target, pred):
    return mean_absolute_error_metric(target, pred, normalize=True)


def r2_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torch.tensor(r2_score(target.float(), pred.float()))


def adj_r2(target, pred, num_params):
    return 1 - (1 - r2_metric(target, pred))*(len(target) - 1)/(len(target) - num_params - 1)
    

def spearman_metric(target, pred):
    import scipy

    target = target.numpy() if torch.is_tensor(target) else target
    pred = pred.numpy() if torch.is_tensor(pred) else pred
    r = scipy.stats.spearmanr(target, pred)
    return torch.tensor(r[0])

from __future__ import annotations
from abc import ABCMeta

import torch
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_pinball_loss,
)

"""
===============================
Quantile Regression
===============================
"""


class QuantileMetric(metaclass=ABCMeta):
    def __init__(self, quantiles=tuple(i / 10 for i in range(1, 10))):
        self.quantiles = quantiles

    @property
    def __name__(self):
        return self.__class__.__name__


class PinballLossMetric(QuantileMetric):
    normalized = False

    def __call__(self, target, pred):
        """
        Pinball loss, which is a weighted variant of the mean absolute error.
        With quantile=0.5, this is the same as the mean absolute error.
        """
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

        if self.normalized:
            pred = (pred - target.min()) / (target.max() - target.min())
            target = (target - target.min()) / (target.max() - target.min())

        assert len(pred.shape) == 2, f"Expected 2D tensor, got {pred.shape} tensor"
        assert pred.shape[1] == len(self.quantiles)

        return sum(
            mean_pinball_loss(target, pred[..., i], alpha=quantile)
            for i, quantile in enumerate(self.quantiles)
        ) / len(self.quantiles)


class NormalizedPinballLossMetric(PinballLossMetric):
    normalized = True


class QuantileCalibrationMetric(QuantileMetric):
    def __call__(self, target, pred):
        """
        Calibration metric, which is the average absolute difference between the empirical quantile and the target quantile.
        This is the absolute difference of the quantile value to the empirical quantile.
        """
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

        assert len(pred.shape) == 2, f"Expected 2D tensor, got {pred.shape} tensor"
        assert pred.shape[1] == len(
            self.quantiles
        ), f"Expected len of {self.quantiles} quantiles, got {pred.shape[1]} quantiles"

        return sum(
            ((target <= pred[..., i]).float().mean() - quantile).abs().item()
            for i, quantile in enumerate(self.quantiles)
        ) / len(self.quantiles)


class SharpnessMetric(QuantileMetric):
    """
    Sharpness metric, which is the mean width of the prediction intervals.
    The lower the sharpness, the better the model.
    """

    normalized = False

    def __call__(self, target, pred):
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

        if self.normalized:
            pred = (pred - target.min()) / (target.max() - target.min())

        assert len(pred.shape) == 2, f"Expected 2D tensor, got {pred.shape} tensor"
        assert pred.shape[1] == len(self.quantiles)

        # we define sharpness as the mean width of the widest considered interval (min quantile / 0, max quantile / -1)
        return (pred[..., -1] - pred[..., 0]).clamp(min=0).mean().item()


class NormalizedSharpnessMetric(SharpnessMetric):
    normalized = True


class QuantileMAEMetric(QuantileMetric):
    normalized = False

    def __call__(self, target, pred):
        """
        Quantile MAE, which is the mean absolute error of the quantile predictions.
        """
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

        if self.normalized:
            pred = (pred - target.min()) / (target.max() - target.min())
            target = (target - target.min()) / (target.max() - target.min())

        assert len(pred.shape) == 2, f"Expected 2D tensor, got {pred.shape} tensor"
        assert pred.shape[1] == len(self.quantiles)
        mid_quantile_index = len(self.quantiles) // 2
        mid_quantile = self.quantiles[
            mid_quantile_index
        ]  # 0.5 for median, 0.9 for 90% CI, etc.
        assert mid_quantile == 0.5, "Quantile MAE is only defined for the median"

        return mean_absolute_error(target, pred[..., mid_quantile_index])


class NormalizedQuantileMAEMetric(QuantileMAEMetric):
    normalized = True


class MeanIntervalScoreMetric(QuantileMetric):
    normalized = False

    def __call__(self, target, pred):
        assert [
            abs(self.quantiles[offset] - (1.0 - self.quantiles[-(offset + 1)])) < 0.0001
            for offset in range(len(self.quantiles) // 2)
        ], "Quantiles must be symmetric around 0.5"

        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

        if self.normalized:
            pred = (pred - target.min()) / (target.max() - target.min())
            target = (target - target.min()) / (target.max() - target.min())

        scores = []
        for interval_ind in range(len(self.quantiles) // 2):
            q_left = pred[..., interval_ind]
            q_right = pred[..., -1 - interval_ind]

            sharpness = (
                (q_right - q_left) * self.quantiles[interval_ind] * 2.0
            )  # the quantile coresponds to alpha/2
            calibration = (
                (q_left - target).clip(min=0) + (target - q_right).clip(min=0)
            ) * 2.0
            scores += [(sharpness + calibration).mean().item()]

        return sum(scores) / len(scores)


class MeanNormalizedIntervalScoreMetric(MeanIntervalScoreMetric):
    normalized = True


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


def spearman_metric(target, pred):
    import scipy

    target = target.numpy() if torch.is_tensor(target) else target
    pred = pred.numpy() if torch.is_tensor(pred) else pred
    r = scipy.stats.spearmanr(target, pred)
    return torch.tensor(r[0])

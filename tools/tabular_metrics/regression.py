from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np


class RegressionBenchmark:
    def __init__(self, X, y, model, use_tensor=False):
        if use_tensor:
            import torch

            assert (
                torch.cuda.is_available()
            ), "CUDA is not available. Ensure you have a compatible GPU and PyTorch installed with CUDA support."
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        self.target = np.array(y) if not isinstance(y, np.ndarray) else y
        self.pred = (
            model.predict(X).cpu().numpy()
            if not isinstance(model.predict(X), np.ndarray)
            else model.predict(X)
        )

        self.dim = X.shape[1] if hasattr(X, "shape") else 1
        self.n = X.shape[0] if hasattr(X, "shape") else len(X)
        self.model_name = model.__class__.__name__

    def mae(self, normalized=False):
        if normalized:
            return mean_absolute_error(self.target, self.pred) / np.mean(self.target)
        return mean_absolute_error(self.target, self.pred)

    def nmae(self):
        return self.mae(normalized=True)

    def rmse(self, normalized=False):
        if normalized:
            return np.sqrt(mean_squared_error(self.target, self.pred)) / np.mean(
                self.target
            )
        return np.sqrt(mean_squared_error(self.target, self.pred))

    def nrmse(self):
        return self.rmse(normalized=True)

    def r2(self):
        return r2_score(self.target, self.pred)

    def adj_r2(self):
        return 1 - (1 - self.r2()) * (self.n - 1) / (self.n - self.dim - 1)

import torch
import numpy as np

from .ridgecv import RidgeGCVTorch


def make_decoder(test_type, device, **kwargs):
    if test_type == 'classify':
        if 'C' not in kwargs:
            kwargs['C'] = 1e3
        if 'max_iter' not in kwargs:
            kwargs['max_iter'] = 10000
        decoder = CuMLLogisticRegression(**kwargs)
    elif test_type.endswith('regress'):
        if 'alphas' not in kwargs:
            kwargs['alphas'] = np.logspace(-8, 8, 17)
        decoder = RidgeGCVTorch(alphas=kwargs['alphas'], device=device)
        decoder = PearsonRScore(decoder)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    pipe = TensorPipeline([
        ('flatten', TensorFlatten()),
        ('scaler', TensorStandardScaler()),
        ('decoder', decoder)
    ])

    return pipe

# --- Tensor-compatible scaler and pipeline ---
class TensorStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, unbiased=False, keepdim=True)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        return (X - self.mean) / self.std, y

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X, y)

class TensorFlatten:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        return X.view(X.shape[0], -1), y.view(y.shape[0], -1) if y is not None else None

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

class TensorPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for name, step in self.steps[:-1]:
            X, y = step.fit_transform(X, y)
        name, step = self.steps[-1]
        if y is not None:
            step.fit(X, y)
        else:
            step.fit(X)
        return self

    def predict(self, X):
        for name, step in self.steps[:-1]:
            X, _ = step.transform(X, None)
        name, step = self.steps[-1]
        return step.predict(X)

    def predict_proba(self, X):
        for name, step in self.steps[:-1]:
            X, _ = step.transform(X, None)
        name, step = self.steps[-1]
        return step.predict_proba(X)

    def score(self, X, y):
        for name, step in self.steps[:-1]:
            X, y = step.transform(X, y)
        name, step = self.steps[-1]
        return step.score(X, y)

class CuMLLogisticRegression:
    def __init__(self, return_tensor=True, **kwargs):
        from cuml.linear_model import LogisticRegression
        self.model = LogisticRegression(**kwargs)
        self.return_tensor = return_tensor  # whether to return torch.Tensor

    def _to_cupy(self, x):
        import cupy as cp
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
        return cp.asarray(x)

    def fit(self, X, y):
        X_cu = self._to_cupy(X)
        y_cu = self._to_cupy(y)
        self.model.fit(X_cu, y_cu)

    def predict(self, X):
        import cupy as cp
        X_cu = self._to_cupy(X)
        preds = self.model.predict(X_cu)
        if self.return_tensor:
            return torch.from_numpy(cp.asnumpy(preds))
        else:
            return cp.asnumpy(preds)
    
    def predict_proba(self, X):
        import cupy as cp
        X_cu = self._to_cupy(X)
        probs = self.model.predict_proba(X_cu)
        if self.return_tensor:
            return torch.from_numpy(cp.asnumpy(probs))
        else:
            return cp.asnumpy(probs)

    def score(self, X, y):
        X_cu = self._to_cupy(X)
        y_cu = self._to_cupy(y)
        score = self.model.score(X_cu, y_cu)
        return float(score)

class PearsonRScore:
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)

    def score(self, X, y):
        import cupy as cp

        pred = self.predict(X)

        # compute in numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        ret = pearsonr(pred, y)
        return ret


def pearsonr(x, y):
    import scipy

    # x, y: (n_samples, n_features)
    xmean = x.mean(axis=0, keepdims=True)
    ymean = y.mean(axis=0, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    normxm = scipy.linalg.norm(xm, axis=0, keepdims=True) + 1e-6
    normym = scipy.linalg.norm(ym, axis=0, keepdims=True) + 1e-6

    r = ((xm / normxm) * (ym / normym)).sum(axis=0)

    return r
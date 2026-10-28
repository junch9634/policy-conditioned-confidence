"""Extracted study functions; use the validated public API where available."""
import numpy as np
from scipy.special import logsumexp
LAMBDA = .001
SCALE_FLOOR = 1e-8

def feature_layout(method):
    if method == "LINEAR_FULL":
        dimension = 4103
        one_hot = np.arange(4096, 4099, dtype=np.int64)
        continuous = np.concatenate((np.arange(0, 4096, dtype=np.int64), np.arange(4099, 4103, dtype=np.int64)))
        source_slice = slice(0, 4103)
    elif method == "LINEAR_OUTPUT7":
        dimension = 7
        one_hot = np.arange(0, 3, dtype=np.int64)
        continuous = np.arange(3, 7, dtype=np.int64)
        source_slice = slice(4096, 4103)
    else:
        raise ValueError(method)
    return dimension, one_hot, continuous, source_slice


def fit_preprocessor(source_features, method):
    dimension, one_hot, continuous, source_slice = feature_layout(method)
    values = np.asarray(source_features[:, source_slice], dtype=np.float64)
    if values.shape[1] != dimension:
        raise RuntimeError(f"Feature dimension mismatch for {method}: {values.shape}")
    mean = values[:, continuous].mean(axis=0)
    std = values[:, continuous].std(axis=0, ddof=0)
    scale = np.maximum(std, SCALE_FLOOR)
    return {
        "method": method,
        "dimension": dimension,
        "one_hot_indices": one_hot,
        "continuous_indices": continuous,
        "source_slice": source_slice,
        "mean": mean,
        "raw_std": std,
        "scale": scale,
    }


def apply_preprocessor(source_features, preprocessor):
    values = np.asarray(source_features[:, preprocessor["source_slice"]], dtype=np.float64).copy()
    continuous = preprocessor["continuous_indices"]
    values[:, continuous] = (values[:, continuous] - preprocessor["mean"]) / preprocessor["scale"]
    return values


def unpack(theta, dimension):
    split = 3 * dimension
    return theta[:split].reshape(3, dimension), theta[split:split + 3]


def objective_and_gradient(theta, x, y):
    dimension = x.shape[1]
    weights, bias = unpack(theta, dimension)
    logits = x @ weights.T + bias
    lse = logsumexp(logits, axis=1)
    ce = float(np.mean(lse - logits[np.arange(len(y)), y]))
    regularization = float(0.5 * LAMBDA * (np.sum(weights * weights) + np.sum(bias * bias)))
    probabilities = np.exp(logits - lse[:, None])
    residual = probabilities
    residual[np.arange(len(y)), y] -= 1.0
    residual /= len(y)
    gradient_weights = residual.T @ x + LAMBDA * weights
    gradient_bias = residual.sum(axis=0) + LAMBDA * bias
    gradient = np.concatenate((gradient_weights.ravel(), gradient_bias))
    return ce + regularization, gradient


def parameter_diagnostics(theta, x, y):
    dimension = x.shape[1]
    weights, bias = unpack(theta, dimension)
    logits = x @ weights.T + bias
    lse = logsumexp(logits, axis=1)
    ce = float(np.mean(lse - logits[np.arange(len(y)), y]))
    regularization = float(0.5 * LAMBDA * (np.sum(weights * weights) + np.sum(bias * bias)))
    objective, gradient = objective_and_gradient(theta.copy(), x, y)
    return {
        "ce": ce,
        "regularization": regularization,
        "objective": float(objective),
        "gradient_inf_norm": float(np.max(np.abs(gradient))),
        "all_finite": bool(np.isfinite(theta).all() and np.isfinite(logits).all() and np.isfinite(objective) and np.isfinite(gradient).all()),
    }


def initial_parameters(y, dimension):
    counts = np.bincount(y, minlength=3).astype(np.float64)
    add_one = (counts + 1.0) / (len(y) + 3.0)
    log_probabilities = np.log(add_one)
    bias = log_probabilities - log_probabilities.mean()
    weights = np.zeros((3, dimension), dtype=np.float64)
    return np.concatenate((weights.ravel(), bias)), counts.astype(np.int64), add_one


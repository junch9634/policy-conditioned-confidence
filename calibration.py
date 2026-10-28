"""Extracted study functions; use the validated public API where available."""
import math
import numpy as np
from scipy.special import logsumexp, expit
LAMBDA = .001
LOWER_A = 1e-6
SCORE_CLIP_EPSILON = 1e-12

def selected_vs_other_logit(logits, selected_index):
    mask = np.arange(len(logits)) != selected_index
    return float(logits[selected_index] - logsumexp(logits[mask]))


def clipped_logit(probability):
    clipped = float(np.clip(probability, SCORE_CLIP_EPSILON, 1.0 - SCORE_CLIP_EPSILON))
    return float(math.log(clipped) - math.log1p(-clipped)), clipped, clipped != probability


def calibration_objective_and_gradient(theta, u, z):
    a, b = float(theta[0]), float(theta[1])
    t = a * u + b
    loss = float(np.mean(np.logaddexp(0.0, t) - z * t))
    regularization = float(0.5 * LAMBDA * (a * a + b * b))
    residual = (expit(t) - z) / len(z)
    gradient = np.asarray([
        float(np.sum(residual * u) + LAMBDA * a),
        float(np.sum(residual) + LAMBDA * b),
    ], dtype=np.float64)
    return loss + regularization, gradient


def calibrator_diagnostics(theta, u, z):
    objective, gradient = calibration_objective_and_gradient(theta.copy(), u, z)
    a, b = float(theta[0]), float(theta[1])
    t = a * u + b
    nll = float(np.mean(np.logaddexp(0.0, t) - z * t))
    regularization = float(0.5 * LAMBDA * (a * a + b * b))
    boundary_active = bool(a <= LOWER_A + 1e-12)
    projected = gradient.copy()
    if boundary_active and gradient[0] > 0.0:
        projected[0] = 0.0
    return {
        "a": a,
        "b": b,
        "binary_nll": nll,
        "regularization": regularization,
        "objective": float(objective),
        "gradient_a": float(gradient[0]),
        "gradient_b": float(gradient[1]),
        "gradient_infinity_norm": float(np.max(np.abs(gradient))),
        "projected_gradient_a": float(projected[0]),
        "projected_gradient_b": float(projected[1]),
        "projected_gradient_infinity_norm": float(np.max(np.abs(projected))),
        "boundary_active": boundary_active,
        "all_finite": bool(np.isfinite(theta).all() and np.isfinite(t).all() and np.isfinite(objective) and np.isfinite(gradient).all()),
    }


def apply_calibrator(score, parameters):
    u = (score - parameters["mu"]) / parameters["scale"]
    t = parameters["a"] * u + parameters["b"]
    return float(u), float(t), float(expit(t))


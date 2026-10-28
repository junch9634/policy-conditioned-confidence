"""Extracted study functions; use the validated public API where available."""
import math
import numpy as np
NLL_CLIP = 1e-7
COVERAGES = (.5, .75, .9, 1.)

def average_rank_auroc(confidence, z):
    n = len(z)
    positives = int(np.sum(z == 1))
    negatives = n - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(confidence, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    position = 0
    while position < n:
        end = position + 1
        while end < n and confidence[order[end]] == confidence[order[position]]:
            end += 1
        average = ((position + 1) + end) / 2.0
        ranks[order[position:end]] = average
        position = end
    rank_sum = float(np.sum(ranks[z == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_tie_aurc(confidence, z):
    order = np.argsort(-confidence, kind="mergesort")
    total_risk = 0.0
    prefix_n = 0
    prefix_errors = 0.0
    position = 0
    while position < len(z):
        end = position + 1
        while end < len(z) and confidence[order[end]] == confidence[order[position]]:
            end += 1
        block = z[order[position:end]]
        block_n = len(block)
        block_errors = float(np.sum(1 - block))
        for taken in range(1, block_n + 1):
            expected_errors = prefix_errors + taken * block_errors / block_n
            total_risk += expected_errors / (prefix_n + taken)
        prefix_n += block_n
        prefix_errors += block_errors
        position = end
    return total_risk / len(z)


def expected_coverage_error(confidence, z, coverage):
    target = int(math.ceil(coverage * len(z)))
    order = np.argsort(-confidence, kind="mergesort")
    selected_n = 0
    expected_errors = 0.0
    position = 0
    while selected_n < target:
        end = position + 1
        while end < len(z) and confidence[order[end]] == confidence[order[position]]:
            end += 1
        block = z[order[position:end]]
        take = min(target - selected_n, len(block))
        expected_errors += take * float(np.sum(1 - block)) / len(block)
        selected_n += take
        position = end
    return expected_errors / target, target


def metric_bundle(confidence, z):
    confidence = np.asarray(confidence, dtype=np.float64)
    z = np.asarray(z, dtype=np.int64)
    clipped = np.clip(confidence, NLL_CLIP, 1.0 - NLL_CLIP)
    result = {
        "n": len(z),
        "correct": int(np.sum(z)),
        "errors": int(np.sum(1 - z)),
        "auroc": average_rank_auroc(confidence, z),
        "aurc_expected_tie_order": expected_tie_aurc(confidence, z),
        "brier": float(np.mean((confidence - z) ** 2)),
        "binary_nll": float(np.mean(-(z * np.log(clipped) + (1 - z) * np.log(1.0 - clipped)))),
    }
    for coverage in COVERAGES:
        error, count = expected_coverage_error(confidence, z, coverage)
        key = f"coverage_{int(coverage * 100)}_error_rate"
        result[key] = error
        result[f"coverage_{int(coverage * 100)}_count"] = count
        result[f"coverage_{int(coverage * 100)}_actual"] = count / len(z)
    return result


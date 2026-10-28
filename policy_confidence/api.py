"""Guarded adapters. No filesystem discovery, fitting, or default study inputs."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import numpy as np

from . import _reference_metrics as metric_reference
from ._reference_cache import load_common_cache

LABELS = ("YES", "NO", "UNKNOWN")


def unique_ids(ids):
    ids = list(ids)
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs")
    return ids


def label_indices(labels):
    labels = list(labels)
    if any(label not in LABELS for label in labels):
        raise ValueError("only resolved YES/NO/UNKNOWN labels are allowed; null is not UNKNOWN")
    return np.asarray([LABELS.index(label) for label in labels], dtype=np.int64)


def align_by_id(ids, rows):
    ids = unique_ids(ids)
    mapping = {}
    for row in rows:
        item = row.get("review_item_id")
        unique_ids([item])
        if item in mapping:
            raise ValueError("duplicate record ID")
        mapping[item] = row
    if set(mapping) != set(ids):
        raise ValueError("record IDs do not match feature IDs")
    return [mapping[item] for item in ids]


def load_features(directory):
    """Strict adapter for the documented three-file cache, not raw study inputs."""
    directory = Path(directory)
    with (directory / "COMMON_FEATURES.npz").open("rb") as handle, np.load(handle, allow_pickle=False) as payload:
        unique_ids(payload["review_item_ids"].tolist())
        for name in ("parse_valid_mask", "first_token_interface_valid_mask", "common_feature_valid_mask"):
            if payload[name].dtype != np.bool_:
                raise ValueError(f"{name} must be a boolean array")
        for name in ("candidate_normalized_probabilities", "ft_raw_confidence"):
            values = payload[name]
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError(f"{name} must contain probabilities in [0,1]")
        if payload["generated_verdict_index"].dtype.kind not in "iu":
            raise ValueError("decision indices must be integers")
    with (directory / "RESPONSE_AND_SCORE_CACHE.jsonl").open(encoding="utf-8") as handle:
        responses = [json.loads(line) for line in handle if line.strip()]
    with (directory / "ID_LINK_MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
        links = list(csv.DictReader(handle))
    unique_ids([row.get("review_item_id") for row in responses])
    unique_ids([row.get("review_item_id") for row in links])
    for row in responses + links:
        for field in ("image_sha256", "target_hash", "policy_hash", "prompt_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"missing or invalid {field}")
    for row in responses:
        if not isinstance(row.get("format_valid"), bool):
            raise ValueError("format_valid must be a boolean")
    return load_common_cache(directory)


def correctness(y_original, resolved_labels):
    """Caller must resolve provenance before passing labels; null is rejected."""
    original = label_indices(y_original)
    gold = label_indices(resolved_labels)
    if original.shape != gold.shape:
        raise ValueError("label arrays have different lengths")
    return (original == gold).astype(np.int64)


def training_priors(y_original_train, y_gold_train):
    """These arguments must contain TRAIN rows only. No test-data fallback."""
    original, gold = label_indices(y_original_train), label_indices(y_gold_train)
    if len(original) == 0 or original.shape != gold.shape:
        raise ValueError("need aligned nonempty train rows")
    gold_count = np.bincount(gold, minlength=3)
    original_count = np.bincount(original, minlength=3)
    correct_count = np.bincount(original[original == gold], minlength=3)
    return {
        "class_order": LABELS,
        "p_gold": (gold_count + 1) / (len(gold) + 3),
        "p_correctness": (correct_count + 1) / (original_count + 2),
    }


def _score_inputs(confidence, z, ranking_score):
    p, raw_z = np.asarray(confidence, dtype=np.float64), np.asarray(z)
    if p.ndim != 1 or raw_z.shape != p.shape:
        raise ValueError("expected aligned one-dimensional probabilities and correctness")
    if any(value is None or value not in (0, 1) for value in raw_z):
        raise ValueError("z must contain only resolved 0/1 values")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("confidence must be finite and in [0,1]")
    scores = p.copy() if ranking_score is None else np.asarray(ranking_score, dtype=np.float64)
    if scores.shape != p.shape or not np.isfinite(scores).all():
        raise ValueError("ranking scores must be aligned and finite")
    return p, raw_z.astype(np.int64), scores


def evaluate_scores(ids, confidence, z, *, ranking_score=None, y_original=None, folds=None):
    """Rank by continuous score; probability errors use confidence."""
    ids = unique_ids(ids)
    p, z, score = _score_inputs(confidence, z, ranking_score)
    if len(ids) != len(p):
        raise ValueError("ID count mismatch")
    names = ("auroc", "aurc", "brier", "binary_nll", "error_at_50", "error_at_100", "within_decision_auroc")
    if len(p) == 0:
        return {"n": 0, **dict.fromkeys(names), "reason": "NO_EVALUABLE_ROWS"}
    result = {
        "n": len(p),
        "auroc": metric_reference.average_rank_auroc(score, z),
        "aurc": metric_reference.expected_tie_aurc(score, z),
        "brier": float(np.mean((p - z) ** 2)),
        "binary_nll": metric_reference.metric_bundle(p, z)["binary_nll"],
        "error_at_50": metric_reference.expected_coverage_error(score, z, .5)[0],
        "error_at_100": metric_reference.expected_coverage_error(score, z, 1.)[0],
        "auroc_na_reason": None if len(set(z)) == 2 else "SINGLE_CORRECTNESS_CLASS",
        "within_decision_auroc": None,
        "within_decision_cells": [],
        "within_decision_weight_sum": 0,
        "within_decision_na_reason": "ORIGINAL_DECISIONS_NOT_PROVIDED",
    }
    if y_original is not None:
        classes = label_indices(y_original)
        if len(classes) != len(p):
            raise ValueError("decision count mismatch")
        fold_values = [None] * len(p) if folds is None else list(folds)
        if len(fold_values) != len(p) or (folds is not None and any(value is None for value in fold_values)):
            raise ValueError("folds must be complete when supplied")
        cells = defaultdict(list)
        for index, key in enumerate(zip(fold_values, classes.tolist())):
            cells[key].append(index)
        numerator, denominator = 0., 0
        for (fold, original), indices in cells.items():
            local = z[indices]
            n_correct, n_wrong = int(local.sum()), int(len(local) - local.sum())
            value = metric_reference.average_rank_auroc(score[indices], local)
            weight = n_correct * n_wrong
            if value is not None:
                numerator += weight * value
                denominator += weight
            result["within_decision_cells"].append({
                "fold": fold, "y_original": LABELS[original], "n": len(local),
                "correct": n_correct, "wrong": n_wrong, "auroc": value,
                "weight": weight, "na_reason": None if value is not None else "SINGLE_CORRECTNESS_CLASS",
            })
        result.update(within_decision_weight_sum=denominator,
                      within_decision_auroc=numerator / denominator if denominator else None,
                      within_decision_na_reason=None if denominator else "NO_VALID_CELLS")
    return result


def compare_pair(positive_confidence, negative_confidence):
    values = [float(positive_confidence), float(negative_confidence)]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("pair confidence must be finite and in [0,1]")
    difference = values[0] - values[1]
    outcome = "WIN" if difference > 0 else "LOSS" if difference < 0 else "TIE"
    return {"difference": difference, "outcome": outcome,
            "concordance": {"WIN": 1., "TIE": .5, "LOSS": 0.}[outcome]}


def check_c_pair(left, right, *, target_relation):
    """Adapter over resolved records, NOT a policy rule engine or GT authoring tool."""
    if target_relation not in ("SAME", "DIFFERENT", "UNDETERMINED"):
        raise ValueError("explicit target relation required")
    unique_ids([left.get("review_item_id"), right.get("review_item_id")])
    for row in (left, right):
        for key in ("image_sha256", "policy_hash"):
            if not row.get(key):
                raise ValueError(f"missing {key}")
        if not isinstance(row.get("decision_parse_valid"), bool):
            raise ValueError("explicit decision_parse_valid boolean required")
    label_valid = [
        row.get("rule_resolution_status") == "RESOLVED" and row.get("evaluation_label") in LABELS
        for row in (left, right)
    ]
    parse_valid = [row["decision_parse_valid"] and row.get("y_original") in LABELS for row in (left, right)]
    z = [
        int(row["evaluation_label"] == row["y_original"]) if lv and pv else None
        for row, lv, pv in zip((left, right), label_valid, parse_valid)
    ]
    checks = {
        "same_image": left["image_sha256"] == right["image_sha256"],
        "same_target": None if target_relation == "UNDETERMINED" else target_relation == "SAME",
        "different_policy": left["policy_hash"] != right["policy_hash"],
        "labels_resolved": True if all(label_valid) else None,
        "both_parse_valid": all(parse_valid),
        "same_generated_decision": left.get("y_original") == right.get("y_original") if all(parse_valid) else None,
        "opposite_correctness": z[0] != z[1] if None not in z else None,
    }
    status = "ELIGIBLE" if all(v is True for v in checks.values()) else (
        "NOT_ELIGIBLE" if any(v is False for v in checks.values()) else "UNRESOLVED")
    return {
        "status": status, "checks": checks, "z": z,
        "failed_conditions": [k for k, v in checks.items() if v is False],
        "unresolved_conditions": [k for k, v in checks.items() if v is None],
        "positive_id": (left if z[0] == 1 else right)["review_item_id"] if status == "ELIGIBLE" else None,
        "negative_id": (right if z[0] == 1 else left)["review_item_id"] if status == "ELIGIBLE" else None,
    }


def matched_b_pairs(c_pairs, fold_number):
    """Exact original degree-one matching, bounded to the studied small graph sizes."""
    from ._reference_ranking import build_b_pairs
    if fold_number not in (1, 2, 3):
        raise ValueError("fold_number must be 1, 2, or 3")
    if not c_pairs:
        raise ValueError("no C pairs; do not fabricate a ranking graph")
    degrees = Counter(item for pair in c_pairs for item in (pair["positive_id"], pair["negative_id"]))
    if any(value != 1 for value in degrees.values()):
        raise ValueError("only degree-one graphs are supported")
    counts = Counter(pair["actual_generated_verdict"] for pair in c_pairs)
    if any(label not in LABELS or count > 3 for label, count in counts.items()):
        raise ValueError("this reference is bounded to at most 3 pairs per decision class")
    if any(not pair.get("budget_group") for pair in c_pairs):
        raise ValueError("explicit budget_group required")
    pairs, report = build_b_pairs(c_pairs, fold_number)
    if not report["raw_degree_match"] or not report["weighted_exposure_match"] or report["duplicate_edge_count"]:
        raise ValueError("exact original matching did not preserve the requested exposure")
    return pairs, report


def loss_terms(method, model, x, z, y_gold, y_original, ids, c_pairs, b_pairs):
    """Compose the original losses; no optimizer is created or executed."""
    import torch
    from ._reference_ranking import compute_losses, stable_original_logit, group_pair_loss
    if method not in ("A", "B", "C", "D", "E", "F"):
        raise ValueError("unknown method")
    ids = unique_ids(ids)
    if len(ids) != len(x) or any(len(values) != len(ids) for values in (z, y_gold, y_original)):
        raise ValueError("training arrays are not aligned")
    index = {item: i for i, item in enumerate(ids)}
    for pair in c_pairs + b_pairs:
        if pair["positive_id"] not in index or pair["negative_id"] not in index:
            raise ValueError("pair endpoint outside supplied TRAIN rows")
        p, n = index[pair["positive_id"]], index[pair["negative_id"]]
        if int(z[p]) != 1 or int(z[n]) != 0 or int(y_original[p]) != int(y_original[n]):
            raise ValueError("ranking direction or generated decision mismatch")
    if not torch.equal(z.long(), (y_gold == y_original).long()):
        raise ValueError("z disagrees with resolved labels and actual generated decisions")
    if method != "F":
        return compute_losses(method, model, x, z, y_gold, y_original, index, c_pairs, b_pairs)
    output = model(x)
    base = torch.nn.functional.cross_entropy(output, y_gold)
    rank = group_pair_loss(stable_original_logit(output, y_original), b_pairs, index)
    return base, rank, base + rank

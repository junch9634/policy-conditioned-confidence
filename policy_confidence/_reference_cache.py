"""Extracted study functions; use the validated public API where available."""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
LABELS = ('YES', 'NO', 'UNKNOWN')

class CacheIntegrationError(ValueError):
    pass


@dataclass
class CommonFeatureBundle:
    review_item_ids: list[str]
    features: np.ndarray
    hidden: np.ndarray
    generated_verdict_indices: np.ndarray
    generated_verdict_onehot: np.ndarray
    candidate_probabilities: np.ndarray
    ft_raw_confidence: np.ndarray
    decision_parse_valid: np.ndarray
    format_valid: np.ndarray
    metadata_by_id: dict[str, dict[str, Any]]
    feature_blocks: dict[str, tuple[int, int]]

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique_map(rows: list[dict[str, Any]], key: str, source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in result:
            raise CacheIntegrationError(f"{source}: duplicate or missing {key}: {value!r}")
        result[value] = row
    return result


def load_common_cache(cache_dir: Path) -> CommonFeatureBundle:
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / "COMMON_FEATURES.npz"
    response_path = cache_dir / "RESPONSE_AND_SCORE_CACHE.jsonl"
    link_path = cache_dir / "ID_LINK_MANIFEST.csv"
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}

    required = {
        "review_item_ids", "candidate_order", "hidden", "generated_verdict_index",
        "generated_verdict_onehot", "candidate_normalized_probabilities", "ft_raw_confidence",
        "parse_valid_mask", "first_token_interface_valid_mask", "common_feature_valid_mask",
    }
    missing_arrays = sorted(required - set(arrays))
    if missing_arrays:
        raise CacheIntegrationError(f"missing NPZ arrays: {missing_arrays}")
    candidate_order = tuple(str(value) for value in arrays["candidate_order"].tolist())
    if candidate_order != LABELS:
        raise CacheIntegrationError(f"candidate order mismatch: {candidate_order}")

    npz_ids = [str(value) for value in arrays["review_item_ids"].tolist()]
    if len(npz_ids) != len(set(npz_ids)):
        raise CacheIntegrationError("duplicate review_item_id in COMMON_FEATURES.npz")
    n = len(npz_ids)
    for name in required - {"candidate_order"}:
        if arrays[name].shape[0] != n:
            raise CacheIntegrationError(f"{name}: row count {arrays[name].shape[0]} != {n}")
    npz_index = {item_id: index for index, item_id in enumerate(npz_ids)}

    response_by_id = _unique_map(_read_jsonl(response_path), "review_item_id", response_path.name)
    with link_path.open(encoding="utf-8", newline="") as handle:
        link_by_id = _unique_map(list(csv.DictReader(handle)), "review_item_id", link_path.name)
    expected = set(npz_ids)
    for source, mapping in ((response_path.name, response_by_id), (link_path.name, link_by_id)):
        if set(mapping) != expected:
            raise CacheIntegrationError(
                f"{source}: ID mismatch missing={sorted(expected - set(mapping))} unexpected={sorted(set(mapping) - expected)}"
            )

    ordered_ids = sorted(expected)
    order = np.asarray([npz_index[item_id] for item_id in ordered_ids], dtype=np.int64)
    hidden = arrays["hidden"][order].astype(np.float32, copy=False)
    onehot = arrays["generated_verdict_onehot"][order].astype(np.float32, copy=False)
    probs = arrays["candidate_normalized_probabilities"][order].astype(np.float32, copy=False)
    ft = arrays["ft_raw_confidence"][order].astype(np.float32, copy=False)
    verdict_indices = arrays["generated_verdict_index"][order].astype(np.int64, copy=False)
    parse_valid = arrays["parse_valid_mask"][order].astype(bool, copy=False)
    interface_valid = arrays["first_token_interface_valid_mask"][order].astype(bool, copy=False)
    common_valid = arrays["common_feature_valid_mask"][order].astype(bool, copy=False)

    if hidden.ndim != 2 or hidden.shape[1] != 4096:
        raise CacheIntegrationError(f"hidden shape is {hidden.shape}, expected (N, 4096)")
    if onehot.shape != (n, 3) or probs.shape != (n, 3) or ft.shape != (n,):
        raise CacheIntegrationError("common feature block shapes are inconsistent")
    if not np.isfinite(hidden).all() or not np.isfinite(onehot).all() or not np.isfinite(probs).all() or not np.isfinite(ft).all():
        raise CacheIntegrationError("NaN/Inf in common feature blocks")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise CacheIntegrationError("candidate-normalized probabilities do not sum to one")
    if not parse_valid.all() or not interface_valid.all() or not common_valid.all():
        raise CacheIntegrationError("this forward-only integration requires all cached feature validity masks to be true")
    if np.any(verdict_indices < 0) or np.any(verdict_indices >= 3):
        raise CacheIntegrationError("invalid generated verdict index")
    expected_onehot = np.eye(3, dtype=np.float32)[verdict_indices]
    if not np.array_equal(onehot, expected_onehot):
        raise CacheIntegrationError("generated verdict one-hot does not match verdict index")
    selected_probs = probs[np.arange(n), verdict_indices]
    if not np.allclose(ft, selected_probs, atol=1e-7):
        raise CacheIntegrationError("FT raw confidence is not probability of the actual generated verdict")

    metadata_by_id: dict[str, dict[str, Any]] = {}
    format_valid = np.zeros(n, dtype=bool)
    for output_index, item_id in enumerate(ordered_ids):
        response = response_by_id[item_id]
        link = link_by_id[item_id]
        source_index = npz_index[item_id]
        expected_label = LABELS[int(arrays["generated_verdict_index"][source_index])]
        if response.get("parsed_verdict") != expected_label or response.get("parse_status") != "PARSED":
            raise CacheIntegrationError(f"{item_id}: parsed verdict mismatch")
        for field in ("input_id", "development_image_id", "image_sha256", "target_hash", "policy_id", "policy_hash", "prompt_sha256"):
            if str(response.get(field, "")) != str(link.get(field, "")):
                raise CacheIntegrationError(f"{item_id}: response/link mismatch for {field}")
        format_valid[output_index] = bool(response.get("format_valid"))
        metadata_by_id[item_id] = {
            "review_item_id": item_id,
            "input_id": response["input_id"],
            "development_image_id": response["development_image_id"],
            "image_sha256": response["image_sha256"],
            "target": response["target"],
            "target_hash": response["target_hash"],
            "policy_id": response["policy_id"],
            "policy_hash": response["policy_hash"],
            "prompt_sha256": response["prompt_sha256"],
            "generated_verdict": response["parsed_verdict"],
            "generated_response": response["generated_text"],
            "decision_parse_valid": True,
            "format_valid": bool(response["format_valid"]),
            "format_error": response.get("format_error"),
            "usage_history_status": response.get("usage_history_status"),
            "source_group_candidate": response.get("source_group_candidate"),
            "source_group_verification_status": response.get("source_group_verification_status"),
            "policy_family_verification_status": response.get("policy_family_verification_status"),
            "policy_priority_review_status": response.get("policy_priority_review_status"),
            "human_gold_in_features": False,
            "policy_role_in_features": False,
        }

    features = np.concatenate((hidden, onehot, probs, ft[:, None]), axis=1).astype(np.float32, copy=False)
    if features.shape != (n, 4103) or not np.isfinite(features).all():
        raise CacheIntegrationError(f"assembled feature matrix invalid: {features.shape}")
    return CommonFeatureBundle(
        review_item_ids=ordered_ids,
        features=features,
        hidden=hidden,
        generated_verdict_indices=verdict_indices,
        generated_verdict_onehot=onehot,
        candidate_probabilities=probs,
        ft_raw_confidence=ft,
        decision_parse_valid=parse_valid,
        format_valid=format_valid,
        metadata_by_id=metadata_by_id,
        feature_blocks={
            "hidden": (0, 4096),
            "generated_verdict_onehot_YES_NO_UNKNOWN": (4096, 4099),
            "candidate_normalized_probabilities_YES_NO_UNKNOWN": (4099, 4102),
            "generated_verdict_ft_raw_confidence": (4102, 4103),
        },
    )


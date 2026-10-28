"""Only fictitious IDs, synthetic arrays, and formula checks. No optimizer steps."""
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.special import softmax

from policy_confidence import api, linear, calibration
from policy_confidence import _reference_metrics as legacy_metrics
from policy_confidence import _reference_ranking as ranking
from policy_confidence.models import make_binary_predictor, make_gold_classifier

ROOT = Path(__file__).resolve().parents[1]


def fake_hash(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def synthetic_cache(path, order=(2, 0, 1), *, bad_prob=False):
    n = 3
    ids = ["SYNTH-A", "SYNTH-B", "SYNTH-C"]
    indices = np.array([0, 1, 2], dtype=np.int64)
    probs = np.array([[.1, .7, .2], [.2, .6, .2], [.1, .2, .7]], dtype=np.float32)
    if bad_prob:
        probs[0] = [-.1, .8, .3]
    hidden = np.arange(n * 4096, dtype=np.float32).reshape(n, 4096) / 4096
    order = np.array(order)
    np.savez(
        path / "COMMON_FEATURES.npz",
        review_item_ids=np.array(ids)[order], candidate_order=np.array(api.LABELS),
        hidden=hidden[order], generated_verdict_index=indices[order],
        generated_verdict_onehot=np.eye(3, dtype=np.float32)[indices][order],
        candidate_normalized_probabilities=probs[order],
        ft_raw_confidence=probs[np.arange(n), indices][order],
        parse_valid_mask=np.ones(n, dtype=bool),
        first_token_interface_valid_mask=np.ones(n, dtype=bool),
        common_feature_valid_mask=np.ones(n, dtype=bool),
    )
    records = []
    for index, item in enumerate(ids):
        records.append({
            "review_item_id": item, "input_id": item, "development_image_id": "SYNTH-IMAGE-" + item,
            "image_sha256": fake_hash(item + "image"), "target": "synthetic target",
            "target_hash": fake_hash("synthetic target"), "policy_id": "SYNTH-POLICY-" + item,
            "policy_hash": fake_hash(item + "policy"), "prompt_sha256": fake_hash(item + "prompt"),
            "parsed_verdict": api.LABELS[index], "parse_status": "PARSED",
            "format_valid": index != 2, "format_error": "SYNTHETIC_FORMAT_ERROR" if index == 2 else None,
            "generated_text": "SYNTHETIC TEXT ONLY", "usage_history_status": "SYNTHETIC",
        })
    (path / "RESPONSE_AND_SCORE_CACHE.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(records)), encoding="utf-8")
    fields = ["review_item_id", "input_id", "development_image_id", "image_sha256",
              "target_hash", "policy_id", "policy_hash", "prompt_sha256"]
    with (path / "ID_LINK_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records[1:] + records[:1])
    return records


def synthetic_c():
    return [
        {"positive_id": "SYNTH-P1", "negative_id": "SYNTH-N1", "budget_group": "BUDGET-A", "actual_generated_verdict": "NO"},
        {"positive_id": "SYNTH-P2", "negative_id": "SYNTH-N2", "budget_group": "BUDGET-B", "actual_generated_verdict": "NO"},
    ]


class FeatureTests(unittest.TestCase):
    def test_permuted_id_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            synthetic_cache(path)
            first = api.load_features(path)
            synthetic_cache(path, (0, 1, 2))
            second = api.load_features(path)
        self.assertEqual(first.review_item_ids, second.review_item_ids)
        np.testing.assert_array_equal(first.features, second.features)
        self.assertEqual(first.features.shape, (3, 4103))
        self.assertFalse(first.format_valid[2])
        self.assertTrue(first.decision_parse_valid[2])
        self.assertEqual(api.LABELS[first.generated_verdict_indices[2]], "UNKNOWN")

    def test_feature_whitelist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            records = synthetic_cache(path)
            first = api.load_features(path)
            for row in records:
                row.update(gold_label="DO_NOT_READ", policy_role="DO_NOT_READ", z="DO_NOT_READ")
            (path / "RESPONSE_AND_SCORE_CACHE.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            second = api.load_features(path)
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.features[:, 4096:4099], np.eye(3))
        self.assertAlmostEqual(float(first.features[0, -1]), .1, places=6)

    def test_out_of_range_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic_cache(Path(directory), bad_prob=True)
            with self.assertRaises(ValueError):
                api.load_features(directory)

    def test_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            records = synthetic_cache(path)
            records[0]["policy_hash"] = fake_hash("different policy")
            (path / "RESPONSE_AND_SCORE_CACHE.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaises(ValueError):
                api.load_features(path)

    def test_duplicate_and_unknown_ids(self):
        with self.assertRaises(ValueError):
            api.align_by_id(["SYNTH-A"], [{"review_item_id": "SYNTH-A"}] * 2)
        with self.assertRaises(ValueError):
            api.align_by_id(["SYNTH-A"], [{"review_item_id": "SYNTH-B"}])
        with self.assertRaises(ValueError):
            api.unique_ids([None])

    def test_unknown_not_null(self):
        np.testing.assert_array_equal(api.correctness(["UNKNOWN"], ["UNKNOWN"]), [1])
        for label in (None, "UNREVIEWED", "UNDETERMINABLE", "DEFERRED"):
            with self.assertRaises(ValueError):
                api.correctness(["UNKNOWN"], [label])

    def test_train_priors(self):
        priors = api.training_priors(["YES", "NO", "NO"], ["YES", "YES", "UNKNOWN"])
        np.testing.assert_allclose(priors["p_gold"], [3/6, 1/6, 2/6])
        np.testing.assert_allclose(priors["p_correctness"], [2/3, 1/4, .5])


class MetricTests(unittest.TestCase):
    def test_exact_ties(self):
        result = api.evaluate_scores(["SYNTH-A", "SYNTH-B"], [.5, .5], [1, 0])
        for metric in ("auroc", "aurc", "error_at_50", "error_at_100"):
            self.assertEqual(result[metric], .5)
        self.assertEqual(result["brier"], .25)
        self.assertAlmostEqual(result["binary_nll"], math.log(2))

    def test_tie_expectation_matches_permutations(self):
        score = np.array([.9, .5, .5, .5, .1])
        z = np.array([1, 0, 1, 0, 1])
        aurcs, errors = [], []
        for permutation in itertools.permutations([1, 2, 3]):
            ordering = [0] + list(permutation) + [4]
            wrong = 1 - z[ordering]
            aurcs.append(float(np.mean(np.cumsum(wrong) / np.arange(1, 6))))
            errors.append(float(wrong[:3].mean()))
        self.assertAlmostEqual(legacy_metrics.expected_tie_aurc(score, z), np.mean(aurcs))
        self.assertAlmostEqual(legacy_metrics.expected_coverage_error(score, z, .5)[0], np.mean(errors))

    def test_undefined_metrics(self):
        result = api.evaluate_scores(["SYNTH-A"], [.2], [1], y_original=["NO"])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["within_decision_auroc"])
        self.assertEqual(result["error_at_100"], 0)
        empty = api.evaluate_scores([], [], [])
        self.assertIsNone(empty["auroc"])
        self.assertEqual(empty["reason"], "NO_EVALUABLE_ROWS")

    def test_reject_invalid_evaluation(self):
        for p, z in (([float("nan")], [1]), ([.5], [None]), ([.5], [2]), ([1.1], [0])):
            with self.assertRaises(ValueError):
                api.evaluate_scores(["SYNTH-A"], p, z)
        with self.assertRaises(ValueError):
            api.evaluate_scores(["SYNTH-A"] * 2, [.1, .2], [1, 0])

    def test_within_decision_weighting(self):
        result = api.evaluate_scores(
            [f"SYNTH-{i}" for i in range(6)],
            [.9, .1, .2, .5, .8, .7], [1, 0, 0, 1, 0, 1],
            y_original=["YES"] * 5 + ["NO"], folds=[1, 1, 1, 2, 2, 2],
        )
        self.assertEqual(result["within_decision_weight_sum"], 3)
        self.assertAlmostEqual(result["within_decision_auroc"], 2/3)

    def test_calibrated_saturation_keeps_original_rank(self):
        ids = ["SYNTH-A", "SYNTH-B"]
        calibrated = api.evaluate_scores(ids, [1., 1.], [1, 0], ranking_score=[100., 99.])
        self.assertEqual(calibrated["auroc"], 1.)
        self.assertEqual(calibrated["error_at_50"], 0.)
        self.assertTrue(math.isfinite(calibrated["binary_nll"]))

    def test_reference_parity(self):
        p = np.array([.8, .5, .5, .2])
        z = np.array([1, 1, 0, 0])
        old = legacy_metrics.metric_bundle(p, z)
        new = api.evaluate_scores([f"SYNTH-{i}" for i in range(4)], p, z)
        for field in ("auroc", "brier", "binary_nll"):
            self.assertEqual(old[field], new[field])
        self.assertEqual(old["aurc_expected_tie_order"], new["aurc"])
        self.assertEqual(old["coverage_50_error_rate"], new["error_at_50"])

    def test_pair_uses_unrounded_exact_tie(self):
        self.assertEqual(api.compare_pair(.5, .5)["outcome"], "TIE")
        self.assertEqual(api.compare_pair(np.nextafter(.5, 1), .5)["outcome"], "WIN")
        self.assertEqual(api.compare_pair(.4, .5)["outcome"], "LOSS")


class PairTests(unittest.TestCase):
    def setUp(self):
        self.left = {"review_item_id": "SYNTH-A", "image_sha256": fake_hash("image"),
                     "policy_hash": fake_hash("policy one"), "rule_resolution_status": "RESOLVED",
                     "evaluation_label": "YES", "y_original": "NO", "decision_parse_valid": True,
                     "format_valid": False}
        self.right = {**self.left, "review_item_id": "SYNTH-B",
                      "policy_hash": fake_hash("policy two"), "evaluation_label": "NO"}

    def test_direction_and_format_separation(self):
        result = api.check_c_pair(self.left, self.right, target_relation="SAME")
        self.assertEqual(result["status"], "ELIGIBLE")
        self.assertEqual(result["positive_id"], "SYNTH-B")
        self.assertEqual(result["negative_id"], "SYNTH-A")

    def test_null_never_wrong(self):
        self.left.update(evaluation_label=None, raw_rule_output="NO", rule_resolution_status="UNRESOLVED")
        result = api.check_c_pair(self.left, self.right, target_relation="SAME")
        self.assertIsNone(result["z"][0])
        self.assertEqual(result["status"], "UNRESOLVED")

    def test_same_unknown_both_wrong(self):
        self.left["y_original"] = self.right["y_original"] = "UNKNOWN"
        result = api.check_c_pair(self.left, self.right, target_relation="SAME")
        self.assertEqual(result["z"], [0, 0])
        self.assertEqual(result["status"], "NOT_ELIGIBLE")

    def test_different_decisions_fail(self):
        self.left["y_original"] = "YES"
        result = api.check_c_pair(self.left, self.right, target_relation="SAME")
        self.assertIn("same_generated_decision", result["failed_conditions"])

    def test_target_not_assumed_from_image(self):
        result = api.check_c_pair(self.left, self.right, target_relation="UNDETERMINED")
        self.assertEqual(result["status"], "UNRESOLVED")

    def test_b_exposure_and_zero_overlap(self):
        c = synthetic_c()
        b, report = api.matched_b_pairs(c, 1)
        self.assertTrue(report["raw_degree_match"])
        self.assertTrue(report["weighted_exposure_match"])
        self.assertEqual(report["C_edge_overlap_count"], 0)
        self.assertEqual(report["duplicate_edge_count"], 0)
        self.assertEqual(ranking.weighted_exposure(b), ranking.weighted_exposure(c))
        self.assertEqual((b, report), api.matched_b_pairs(c, 1))

    def test_no_silent_graph_relaxation(self):
        with self.assertRaises(ValueError):
            api.matched_b_pairs([], 1)
        c = synthetic_c()
        c[1]["positive_id"] = c[0]["positive_id"]
        with self.assertRaises(ValueError):
            api.matched_b_pairs(c, 1)
        _, report = api.matched_b_pairs(synthetic_c()[:1], 1)
        self.assertEqual(report["C_edge_overlap_count"], 1)


class HeadTests(unittest.TestCase):
    def test_architecture_and_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in make_gold_classifier(4103).parameters()), 262851)
        self.assertEqual(sum(p.numel() for p in make_gold_classifier(7).parameters()), 707)
        self.assertEqual(sum(p.numel() for p in make_binary_predictor(4103).parameters()), 262721)
        self.assertEqual(4103 * 3 + 3, 12312)
        self.assertEqual(7 * 3 + 3, 24)

    def test_original_not_argmax(self):
        logits = torch.tensor([[4., 1., -1.], [1., 2., 3.]], dtype=torch.float64)
        original = torch.tensor([1, 0])
        q = torch.softmax(logits, 1)
        selected = q[torch.arange(2), original]
        self.assertTrue(torch.all(selected < q.max(1).values))
        torch.testing.assert_close(torch.sigmoid(ranking.stable_original_logit(logits, original)), selected)
        model = torch.nn.Identity()
        np.testing.assert_allclose(ranking.predict_confidence("F", model, logits, original), selected.numpy())

    def test_all_losses_finite_without_optimizer_or_parameter_updates(self):
        ids = ["SYNTH-P1", "SYNTH-N1", "SYNTH-P2", "SYNTH-N2"]
        x = torch.arange(28, dtype=torch.float64).reshape(4, 7) / 28
        original = torch.ones(4, dtype=torch.long)
        gold = torch.tensor([1, 0, 1, 2])
        z = (gold == original).double()
        c = synthetic_c()
        b, _ = api.matched_b_pairs(c, 1)
        torch.manual_seed(123)
        scalar = make_binary_predictor(7).double()
        classifier = make_gold_classifier(7).double()
        outcomes = {}
        with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer forbidden")):
            for method in "ABCDEF":
                model = copy.deepcopy(scalar if method in "ABC" else classifier)
                before = [p.detach().clone() for p in model.parameters()]
                base, rank, total = api.loss_terms(method, model, x, z, gold, original, ids, c, b)
                self.assertTrue(torch.isfinite(total))
                total.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
                for expected, actual in zip(before, model.parameters()):
                    torch.testing.assert_close(expected, actual)
                outcomes[method] = (base.item(), rank.item(), total.item())
        self.assertEqual(outcomes["D"][1], 0)
        self.assertEqual(outcomes["D"][0], outcomes["E"][0])
        self.assertEqual(outcomes["D"][0], outcomes["F"][0])
        self.assertAlmostEqual(outcomes["F"][2], outcomes["F"][0] + outcomes["F"][1])

    def test_pair_endpoint_leak_is_error(self):
        ids = ["SYNTH-A", "SYNTH-B"]
        model = make_gold_classifier(7)
        x = torch.zeros(2, 7)
        gold, original = torch.tensor([1, 0]), torch.tensor([1, 1])
        with self.assertRaises(ValueError):
            api.loss_terms("E", model, x, torch.tensor([1., 0.]), gold, original, ids, synthetic_c(), [])


class NumericalTests(unittest.TestCase):
    def test_shared_seven_preprocessing(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(6, 4103))
        x[:, 4096:4099] = np.eye(3)[np.arange(6) % 3]
        x[:, 1] = 5.
        full = linear.fit_preprocessor(x, "LINEAR_FULL")
        short = linear.fit_preprocessor(x, "LINEAR_OUTPUT7")
        np.testing.assert_allclose(linear.apply_preprocessor(x, full)[:, 4096:],
                                   linear.apply_preprocessor(x, short))
        np.testing.assert_array_equal(linear.apply_preprocessor(x, full)[:, 4096:4099], x[:, 4096:4099])
        heldout = x[:1].copy() + 100
        before = full["mean"].copy()
        linear.apply_preprocessor(heldout, full)
        np.testing.assert_array_equal(full["mean"], before)
        self.assertEqual(full["scale"][1], 1e-8)

    def test_initial_add_one(self):
        y = np.array([0, 0, 1])
        theta, _, add_one = linear.initial_parameters(y, 7)
        weights, bias = linear.unpack(theta, 7)
        np.testing.assert_array_equal(weights, np.zeros((3, 7)))
        np.testing.assert_allclose(softmax(bias), add_one)
        np.testing.assert_allclose(add_one, [3/6, 2/6, 1/6])

    def test_linear_analytic_gradient(self):
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=(4, 7)), np.array([0, 1, 2, 1])
        theta = rng.normal(scale=.1, size=24)
        _, analytic = linear.objective_and_gradient(theta, x, y)
        eps = 1e-6
        numeric = []
        for i in range(len(theta)):
            direction = np.zeros_like(theta)
            direction[i] = eps
            numeric.append((linear.objective_and_gradient(theta + direction, x, y)[0]
                            - linear.objective_and_gradient(theta - direction, x, y)[0]) / (2 * eps))
        np.testing.assert_allclose(analytic, numeric, atol=2e-9)
        self.assertTrue(linear.parameter_diagnostics(theta, x, y)["all_finite"])

    def test_calibration_analytic_gradient(self):
        theta, u, z = np.array([.7, -.2]), np.array([-2., 0., 1.]), np.array([0., 1., 1.])
        _, analytic = calibration.calibration_objective_and_gradient(theta, u, z)
        eps = 1e-6
        for i in range(2):
            direction = np.zeros(2)
            direction[i] = eps
            numeric = (calibration.calibration_objective_and_gradient(theta + direction, u, z)[0]
                       - calibration.calibration_objective_and_gradient(theta - direction, u, z)[0]) / (2 * eps)
            self.assertAlmostEqual(analytic[i], numeric, places=8)

    def test_projected_gradient_and_monotonicity(self):
        theta = np.array([1e-6, 0.])
        result = calibration.calibrator_diagnostics(theta, np.array([-1., 1.]), np.array([1., 0.]))
        self.assertGreater(result["gradient_a"], 0)
        self.assertEqual(result["projected_gradient_a"], 0)
        parameters = {"mu": 0., "scale": 2., "a": .1, "b": -.2}
        outputs = [calibration.apply_calibrator(x, parameters)[2] for x in [-2., 0., 2.]]
        self.assertEqual(outputs, sorted(outputs))

    def test_logit_recovery(self):
        logits = np.array([1000., 999., -1000.])
        score = calibration.selected_vs_other_logit(logits, 1)
        self.assertTrue(math.isfinite(score))
        self.assertAlmostEqual(score, -1.)
        _, clipped, changed = calibration.clipped_logit(1.)
        self.assertTrue(changed)
        self.assertEqual(clipped, 1 - 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)

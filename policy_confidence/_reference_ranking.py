"""Extracted study functions; use the validated public API where available."""
import itertools
import random
from collections import Counter, defaultdict
from fractions import Fraction
import torch
PAIR_SEED = 20261001
LAMBDA_PAIR = 1.0

def stable_original_logit(logits, original_indices):
    selected = logits.gather(1, original_indices[:, None])[:, 0]
    mask = torch.nn.functional.one_hot(original_indices, num_classes=logits.shape[1]).bool()
    others = logits.masked_fill(mask, float("-inf"))
    return selected - torch.logsumexp(others, dim=1)


def group_pair_loss(scores, pairs, index):
    grouped = defaultdict(list)
    for pair in pairs:
        positive = scores[index[pair["positive_id"]]]
        negative = scores[index[pair["negative_id"]]]
        grouped[pair["budget_group"]].append(torch.nn.functional.softplus(negative - positive))
    if not grouped:
        return scores.sum() * 0.0
    return torch.stack([torch.stack(losses).mean() for losses in grouped.values()]).mean()


def weighted_exposure(pairs):
    group_sizes = Counter(pair["budget_group"] for pair in pairs)
    group_count = len(group_sizes)
    result = defaultdict(Fraction)
    for pair in pairs:
        weight = Fraction(1, group_count * group_sizes[pair["budget_group"]])
        result[pair["positive_id"]] += weight
        result[pair["negative_id"]] += weight
    return dict(result)


def serialize_exposure(exposure):
    return {key: {"fraction": f"{value.numerator}/{value.denominator}", "float": float(value)} for key, value in sorted(exposure.items())}


def build_b_pairs(c_pairs, fold_number):
    c_edges = {(pair["positive_id"], pair["negative_id"]) for pair in c_pairs}
    degree = Counter(endpoint for pair in c_pairs for endpoint in (pair["positive_id"], pair["negative_id"]))
    if any(value != 1 for value in degree.values()):
        raise RuntimeError(f"Fold {fold_number}: observed C graph is not degree one")
    by_class = defaultdict(list)
    for pair in c_pairs:
        by_class[pair["actual_generated_verdict"]].append(pair)
    rng = random.Random(PAIR_SEED + fold_number - 1)
    selected = []
    class_search = {}
    for verdict in sorted(by_class):
        positives = sorted(pair["positive_id"] for pair in by_class[verdict])
        negatives = sorted(pair["negative_id"] for pair in by_class[verdict])
        candidates = []
        for permutation in itertools.permutations(negatives):
            edges = tuple(zip(positives, permutation))
            overlap = sum(edge in c_edges for edge in edges)
            candidates.append((overlap, edges))
        minimum = min(row[0] for row in candidates)
        minimizers = [row[1] for row in candidates if row[0] == minimum]
        chosen = minimizers[rng.randrange(len(minimizers))]
        selected.extend((positive, negative, verdict) for positive, negative in chosen)
        class_search[verdict] = {
            "pair_count": len(positives),
            "enumerated_permutations": len(candidates),
            "minimum_overlap": minimum,
            "minimum_solutions": len(minimizers),
        }
    selected.sort()
    slots = sorted(pair["budget_group"] for pair in c_pairs)
    rng.shuffle(slots)
    pairs = []
    for index, ((positive, negative, verdict), group) in enumerate(zip(selected, slots), 1):
        pairs.append({
            "pair_id": f"B-{index:03d}",
            "positive_id": positive,
            "negative_id": negative,
            "z_positive": 1,
            "z_negative": 0,
            "actual_generated_verdict": verdict,
            "budget_group": group,
            "source": "FOLD_DEGREE_MATCHED_MINIMUM_OVERLAP_CONTROL",
            "budget_group_note": "Loss budget only; not the random edge's physical image/source.",
        })
    b_edges = {(pair["positive_id"], pair["negative_id"]) for pair in pairs}
    b_degree = Counter(endpoint for pair in pairs for endpoint in (pair["positive_id"], pair["negative_id"]))
    report = {
        "pair_seed": PAIR_SEED + fold_number - 1,
        "class_search": class_search,
        "pair_count": len(pairs),
        "raw_degree_match": b_degree == degree,
        "weighted_exposure_match": weighted_exposure(pairs) == weighted_exposure(c_pairs),
        "duplicate_edge_count": len(pairs) - len(b_edges),
        "C_edge_overlap_count": len(b_edges & c_edges),
        "C_edge_overlap_ratio": len(b_edges & c_edges) / len(c_pairs),
        "minimum_overlap_exact_for_observed_degree_one_graph": sum(row["minimum_overlap"] for row in class_search.values()),
    }
    return pairs, report


def compute_losses(method, model, x, z, y_gold, y_original, index, c_pairs, b_pairs):
    output = model(x)
    if method in {"A", "B", "C"}:
        score = output[:, 0]
        base = torch.nn.functional.binary_cross_entropy_with_logits(score, z)
    else:
        base = torch.nn.functional.cross_entropy(output, y_gold)
        score = stable_original_logit(output, y_original)
    pairs = b_pairs if method == "B" else c_pairs if method in {"C", "E"} else []
    rank = group_pair_loss(score, pairs, index)
    return base, rank, base + LAMBDA_PAIR * rank


def predict_confidence(method, model, x, y_original):
    model.eval()
    with torch.no_grad():
        output = model(x)
        if method in {"A", "B", "C"}:
            confidence = torch.sigmoid(output[:, 0])
        else:
            confidence = torch.softmax(output, dim=1).gather(1, y_original[:, None])[:, 0]
    if not torch.isfinite(confidence).all() or not ((confidence >= 0) & (confidence <= 1)).all():
        raise RuntimeError(f"Invalid confidence from {method}")
    return confidence.cpu().numpy()


"""Frozen H3 analysis gates; never imported by official inference."""
import math


def gold_gate(base_dense, base_hybrid, small_dense, small_hybrid, policy):
    n = base_hybrid["queries"]
    if any(row["queries"] != n for row in [base_dense, small_dense, small_hybrid]):
        raise ValueError("Gold comparison query counts differ")
    delta = {key: base_hybrid[key] - small_hybrid[key] for key in
             ["gold_hit@10", "gold_mrr@10", "gold_hit@50"]}
    dense_delta = {key: base_dense[key] - small_dense[key] for key in ["gold_hit@10", "gold_mrr@10"]}
    checks = {"hybrid_hit10_gain": delta["gold_hit@10"] * n >= policy["hybrid_minimum_hit10_count_gain"] - 1e-9,
        "hybrid_mrr10_gain": delta["gold_mrr@10"] >= policy["hybrid_minimum_mrr10_gain"] - 1e-12,
        "hybrid_hit50_nondecreasing": delta["gold_hit@50"] >= -1e-12,
        "dense_hit10_nondecreasing": dense_delta["gold_hit@10"] >= -1e-12,
        "dense_mrr10_nondecreasing": dense_delta["gold_mrr@10"] >= -1e-12}
    return {"passed": all(checks.values()), "checks": checks, "hybrid_differences": delta,
            "dense_differences": dense_delta}


def adoption_gate(base, small5, small6, base_categories, small_categories, policy):
    checks = {}
    for name, small in [("small5", small5), ("small6", small6)]:
        checks[f"recall_vs_{name}"] = base["graded_recall@10"] >= small["graded_recall@10"] - 1e-12
        checks[f"mrr_vs_{name}"] = base["graded_mrr@10"] - small["graded_mrr@10"] >= (
            policy["minimum_mrr10_gain_vs_small5_and_small6"] - 1e-12)
    if set(base_categories) != set(small_categories):
        raise ValueError("category comparison coverage differs")
    for name in base_categories:
        for metric, limit in [("graded_recall@10", policy["maximum_category_recall_drop_vs_small5"]),
                              ("graded_mrr@10", policy["maximum_category_mrr_drop_vs_small5"])]:
            checks[f"{name}:{metric}"] = base_categories[name][metric] >= small_categories[name][metric] - limit - 1e-12
    return {"passed": all(checks.values()), "checks": checks}


def paired_bootstrap(differences, seed=42, samples=10000):
    """Exploratory qid-paired CI, not an independent-test or repeated-seed claim."""
    import numpy as np
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid paired differences")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 256):
        indices = rng.integers(0, len(values), size=(min(256, samples - start), len(values)))
        means.extend(values[indices].mean(axis=1))
    low, high = np.quantile(means, [.025, .975])
    return {"mean_difference": float(values.mean()), "ci95": [float(low), float(high)],
            "samples": samples, "seed": seed, "unit": "paired qid; exploratory on reused dev"}

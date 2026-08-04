import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def normalize_pruning(t):
    mapping = {
        "random": "random_sample",
        "most_similar": "prototype",
        "most_dissimilar": "most_different",
        "decoupled": "decoupled",
        "least_perplexity": "least_perplexity",
        "zipf": "zipf",
        "unique_tokens": "unique_tokens",
        "words_dataset": "words_dataset",
    }
    return mapping.get(t, t)


def main():
    theory_path = "results/theory_empirical/theory_empirical_results.csv"
    eval_path = "results/experiment_results_new.csv"

    th = pd.read_csv(theory_path)
    ev = pd.read_csv(eval_path)

    # normalize column names
    th = th.rename(columns={"technique": "pruning_type", "dataset": "task"})
    th["pruning_type_norm"] = th["pruning_type"].astype(str)

    # normalize eval pruning_type -> theory technique names
    ev["pruning_type_norm"] = ev["pruning_type"].astype(str).apply(normalize_pruning)

    # focus on main metrics (accuracy / exact_match)
    ev_main = ev[ev["metric"].str.contains("acc|exact_match", na=False)]

    # aggregate eval by model,task,pruning_type_norm,nsamples averaging numeric values
    agg = (
        ev_main.groupby(["model", "task", "pruning_type_norm", "nsamples"])
        .agg({"original_value": "mean", "pruned_value": "mean"})
        .reset_index()
    )
    agg["delta"] = agg["pruned_value"] - agg["original_value"]

    # merge with theory metrics
    merged = pd.merge(
        th,
        agg,
        how="inner",
        left_on=["model", "task", "pruning_type", "nsamples"],
        right_on=["model", "task", "pruning_type_norm", "nsamples"],
    )

    if merged.empty:
        print("No merged rows found — check matching keys (model/task/pruning_type/nsamples).")
        return

    metrics = [
        "kl_target_to_calib",
        "kl_calib_to_target",
        "js_divergence",
        "calib_entropy",
        "unique_token_coverage",
        "tail_token_coverage",
        "activation_wasserstein",
        "activation_ks",
    ]

    results = []
    for m in metrics:
        if m not in merged.columns:
            continue
        x = merged[m].astype(float).values
        y = merged["delta"].astype(float).values
        # drop nan pairs
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            continue
        pr, pr_p = pearsonr(x[mask], y[mask])
        sr, sr_p = spearmanr(x[mask], y[mask])
        results.append((m, pr, pr_p, sr, sr_p, mask.sum()))

    resdf = pd.DataFrame(
        results,
        columns=["metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n"],
    )

    out = "results/theory_empirical/correlations_pruned_delta.csv"
    resdf.to_csv(out, index=False)
    print(f"Saved correlation results to {out}")


if __name__ == "__main__":
    main()

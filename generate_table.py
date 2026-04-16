#!/usr/bin/env python3
"""
Generate a LaTeX comparison table: COLA (baseline) vs. a selected sampling
strategy.  The table is oriented with **evaluation tasks as rows** and
**calibration-dataset groups as columns**, making it longer and narrower –
better suited for a two-column paper layout.

Usage example
─────────────
    python3 generate_table.py \
        --compression_type pruning \
        --pruning_type words_dataset \
        --model meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b-it

Calibration-dataset groups
──────────────────────────
  (i)   Language Modeling         : wikitext, c4, pile
  (ii)  Mathematical Reasoning    : gsm8k, svamp
  (iii) Commonsense Reasoning & QA: winogrande, openbookqa
  (iv)  NLI                       : rte, anli_r1
  (v)   Knowledge & Translation   : mmlu, wmt14

Highlighting
────────────
  • Per-task rows: top-3 values across COLA + Ours calib-group columns
    are highlighted with \\ulc{value}{foresta} (1st), foresta!70 (2nd),
    foresta!40 (3rd).
  • Mean / Geo% columns: the higher value between COLA and Ours is bolded.
"""

import argparse
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
COLA_CSV      = SCRIPT_DIR / "results" / "cola_experiments_multilingual.csv"
EXP_CSV       = SCRIPT_DIR / "results" / "experiments_multilingual.csv"
COLA_2SSP_CSV = SCRIPT_DIR / "results" / "2ssp_cola_experiment_results.csv"
EXP_2SSP_CSV  = SCRIPT_DIR / "results" / "2ssp_experiment_results.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
METRIC_ACC_NORM = "acc_norm,none"
METRIC_ACC      = "acc,none"
METRIC_GSM8K    = "exact_match,flexible-extract"
METRIC_XQUAD      = "f1,none"

TASKS_WITH_NORM = {"arc_challenge", "arc_easy", "hellaswag", "openbookqa"}

# Evaluation tasks (rows)
EVAL_TASKS = [
    # "arc_challenge", "arc_easy", "boolq", "hellaswag",
    # "winogrande", "gsm8k", "mmlu", "openbookqa", "rte", "anli_r1",
    "global_mmlu_es", "global_mmlu_zh", "xquad_es", "xquad_zh",
    "xnli_es", "xnli_zh", "xwinograd_zh",
    "xcopa_zh",
]

TASK_DISPLAY = {
    # "arc_challenge": "ARC-C",
    # "arc_easy":      "ARC-E",
    # "boolq":         "BoolQ",
    # "hellaswag":     "HellaSwag",
    # "winogrande":    "WinoGr.",
    # "gsm8k":         "GSM8k",
    # "mmlu":          "MMLU",
    # "openbookqa":    "OBQA",
    # "rte":           "RTE",
    # "anli_r1":       "ANLI",
    "global_mmlu_es":     "MMLU-ES",
    "global_mmlu_zh":     "MMLU-ZH",
    "xquad_es":          "XQuAD-ES",
    "xquad_zh":          "XQuAD-ZH",
    "xnli_es":           "XNLI-ES",
    "xnli_zh":           "XNLI-ZH",
    "xwinograd_zh":       "XWino-ZH",
    "xcopa_zh":          "XCOPA-ZH",
}

# Calibration-dataset groups (columns)
CALIB_GROUPS = OrderedDict([
    # ("(i)",   ["wikitext", "c4", "pile"]),
    # ("(ii)",  ["gsm8k", "svamp"]),
    # ("(iii)", ["winogrande", "openbookqa"]),
    # ("(iv)",  ["rte", "anli_r1"]),
    # ("(v)",   ["mmlu", "wmt14"]),
    # Multilingual Splits
    ("(Know-es)", ["global_mmlu_es", "xnli_es", "xquad_es"]),
    ("(Know-zh)", ["global_mmlu_zh", "xnli_zh", "xquad_zh"]),
    ("(Comm-zh)", ["xwinograd_zh", "xcopa_zh"]),
])
N_GROUPS = len(CALIB_GROUPS)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _metric_for(task: str) -> str:
    if task == "gsm8k":
        return METRIC_GSM8K
    if task in {"xquad_es", "xquad_zh"}:
        return METRIC_XQUAD
    return METRIC_ACC_NORM if task in TASKS_WITH_NORM else METRIC_ACC


def _escape(s: str) -> str:
    for ch in ("%", "#", "$"):
        s = s.replace(ch, "\\" + ch)
    s = s.replace("_", r"\_")
    return s


def _fmt(v, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if pct:
        return f"{v:.1f}"
    return f"{v:.4f}"


def _ulc(formatted: str, rank: int) -> str:
    """Wrap a formatted value in \\ulc{...}{color} for top-3 highlighting."""
    color = {0: "foresta", 1: "foresta!70", 2: "foresta!40"}[rank]
    return r"\ulc{" + formatted + "}{" + color + "}"


# ── Data loading ──────────────────────────────────────────────────────────────

def _dense_values(exp_df: pd.DataFrame, model: str) -> dict:
    """Original (dense) score per evaluation task."""
    out = {}
    for task in EVAL_TASKS:
        metric = _metric_for(task)
        rows = exp_df.loc[
            (exp_df["model"] == model)
            & (exp_df["task"] == task)
            & (exp_df["metric"] == metric)
        ]
        out[task] = float(rows["original_value"].iloc[0]) if not rows.empty else float("nan")
    return out


def _cola_values(
    cola_df: pd.DataFrame,
    model: str,
    compression_type: str,
    calib_datasets: list,
) -> dict:
    """COLA score per eval task, averaged over *calib_datasets*."""
    out = {}
    for task in EVAL_TASKS:
        metric = _metric_for(task)
        rows = cola_df.loc[
            (cola_df["model"] == model)
            & (cola_df["task"] == task)
            & (cola_df["metric"] == metric)
            & (cola_df["pruning_type"] == compression_type)
            & (cola_df["datasets"].isin(calib_datasets))
        ]
        out[task] = float(rows["value"].mean()) if not rows.empty else float("nan")
    return out


def _exp_values(
    exp_df: pd.DataFrame,
    model: str,
    compression_type: str,
    pruning_type: str,
    calib_datasets: list,
) -> dict:
    """Our-method score per eval task, averaged over *calib_datasets*."""
    out = {}
    for task in EVAL_TASKS:
        metric = _metric_for(task)
        rows = exp_df.loc[
            (exp_df["model"] == model)
            & (exp_df["task"] == task)
            & (exp_df["metric"] == metric)
            & (exp_df["compression_type"] == compression_type)
            & (exp_df["pruning_type"] == pruning_type)
            & (exp_df["calibration_datasets"].isin(calib_datasets))
        ]
        out[task] = float(rows["pruned_value"].mean()) if not rows.empty else float("nan")
    return out


# ── Aggregate stats ───────────────────────────────────────────────────────────

def _arith_mean_list(vals: list) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def _geo_mean_pct_list(vals: list, dense_vals: list) -> float:
    """Geometric mean of (value / dense) ratios × 100."""
    ratios = []
    for p, d in zip(vals, dense_vals):
        if not math.isnan(p) and not math.isnan(d) and d > 0 and p > 0:
            ratios.append(p / d)
    if not ratios:
        return float("nan")
    return math.exp(np.mean([math.log(r) for r in ratios])) * 100


# ── Top-3 logic ──────────────────────────────────────────────────────────────

def _top3_indices(values: list) -> list:
    """Return indices of the top-3 values (descending).  NaN is ignored."""
    indexed = [(i, v) for i, v in enumerate(values) if not math.isnan(v)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in indexed[:3]]


# ── LaTeX generation ──────────────────────────────────────────────────────────

def _build_latex(
    models: list,
    compression_type: str,
    pruning_type: str,
    cola_df: pd.DataFrame,
    exp_df: pd.DataFrame,
) -> str:
    """
    Build the swapped table:
      Rows  = eval tasks  (+Mean, +Geo%)   per model
      Cols  = Dense | COLA (i)–(v) Mean Geo% | Ours (i)–(v) Mean Geo%
    """
    group_keys = list(CALIB_GROUPS.keys())  # (i)–(v)
    n_g       = N_GROUPS                    # 5

    # Column layout: Model | Task | Dense || COLA×5 | Mean | Geo% || Ours×5 | Mean | Geo%
    #   = 2 label cols + 1 dense + 7 COLA + 7 Ours = 17 total
    n_cola_cols = n_g + 2    # 7
    n_ours_cols = n_g + 2    # 7
    col_spec = (
        "l|l||c||"
        + "c" * n_g + "|c|c||"
        + "c" * n_g + "|c|c"
    )

    ours_label = _escape(pruning_type)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # ── Header row 1: macro groups ───────────────────────────────────────────
    cola_start = 4                            # Dense is col 3
    cola_end   = cola_start + n_cola_cols - 1
    ours_start = cola_end + 1
    ours_end   = ours_start + n_ours_cols - 1
    lines.append(
        r" &  & "
        + r" & \multicolumn{" + str(n_cola_cols) + r"}{c||}{\textbf{COLA}}"
        + r" & \multicolumn{" + str(n_ours_cols) + r"}{c}{\textbf{" + ours_label + r"}} \\"
    )
    lines.append(
        r"\cmidrule(lr){" + str(cola_start) + "-" + str(cola_end) + "} "
        r"\cmidrule(lr){" + str(ours_start) + "-" + str(ours_end) + "}"
    )

    # ── Header row 2: sub-columns ────────────────────────────────────────────
    grp_hdrs = " & ".join(g for g in group_keys)
    lines.append(
        r"\textbf{Model} & \textbf{Task} & \textbf{Dense}"
        + r" & " + grp_hdrs + r" & \textbf{Mean} & \textbf{Geo\%}"
        + r" & " + grp_hdrs + r" & \textbf{Mean} & \textbf{Geo\%} \\"
    )
    lines.append(r"\midrule")

    # ── Data rows per model ──────────────────────────────────────────────────
    for mi, model in enumerate(models):
        model_short = model.split("/")[-1]
        dense = _dense_values(exp_df, model)

        # Pre-compute all group-level values for this model
        # cola_grid[group_key][task]  and  ours_grid[group_key][task]
        cola_grid = {}
        ours_grid = {}
        for gk, calib_ds in CALIB_GROUPS.items():
            cola_grid[gk] = _cola_values(cola_df, model, compression_type, calib_ds)
            ours_grid[gk] = _exp_values(exp_df, model, compression_type, pruning_type, calib_ds)

        n_task_rows = len(EVAL_TASKS)
        n_all_rows  = n_task_rows + 2  # +Mean +Geo%
        model_cell  = (
            r"\multirow{"
            + str(n_all_rows)
            + r"}{*}{\rotatebox[origin=c]{90}{\textbf{"
            + _escape(model_short)
            + r"}}}"
        )

        # ─ Per-task rows ─────────────────────────────────────────────────────
        # Collect column-wise lists for bottom summary
        # col_vals_cola[gk] = [val_task0, val_task1, ...] etc.
        col_vals_cola = {gk: [] for gk in group_keys}
        col_vals_ours = {gk: [] for gk in group_keys}
        dense_list    = []

        for ti, task in enumerate(EVAL_TASKS):
            d_val = dense[task]
            dense_list.append(d_val)

            # collect the 5 COLA + 5 Ours values for this task
            c_vals = [cola_grid[gk][task] for gk in group_keys]
            o_vals = [ours_grid[gk][task] for gk in group_keys]

            for gk, cv in zip(group_keys, c_vals):
                col_vals_cola[gk].append(cv)
            for gk, ov in zip(group_keys, o_vals):
                col_vals_ours[gk].append(ov)

            # Mean and Geo% across groups for this task
            c_mean = _arith_mean_list(c_vals)
            o_mean = _arith_mean_list(o_vals)
            c_geo  = _geo_mean_pct_list(c_vals, [d_val] * n_g)
            o_geo  = _geo_mean_pct_list(o_vals, [d_val] * n_g)

            # ── top-3 among the 10 group values (5 COLA + 5 Ours) ───────────
            all_group_vals = c_vals + o_vals  # length 10
            top3 = _top3_indices(all_group_vals)  # indices 0-9

            # Format the 10 values, applying \ulc for top-3
            all_fmt = [_fmt(v) for v in all_group_vals]
            for rank, idx in enumerate(top3):
                if all_fmt[idx] != "--":
                    all_fmt[idx] = _ulc(all_fmt[idx], rank)

            c_cells = " & ".join(all_fmt[:n_g])
            o_cells = " & ".join(all_fmt[n_g:])

            # ── format Mean / Geo%: bold the winner ──────────────────────────
            c_mean_s = _fmt(c_mean)
            o_mean_s = _fmt(o_mean)
            c_geo_s  = _fmt(c_geo, pct=True)
            o_geo_s  = _fmt(o_geo, pct=True)

            if not math.isnan(c_mean) and not math.isnan(o_mean):
                if o_mean >= c_mean:
                    o_mean_s = r"\textbf{" + o_mean_s + "}"
                else:
                    c_mean_s = r"\textbf{" + c_mean_s + "}"
            if not math.isnan(c_geo) and not math.isnan(o_geo):
                if o_geo >= c_geo:
                    o_geo_s = r"\textbf{" + o_geo_s + "}"
                else:
                    c_geo_s = r"\textbf{" + c_geo_s + "}"

            label = model_cell if ti == 0 else ""
            task_name = TASK_DISPLAY.get(task, task)

            lines.append(
                label
                + " & " + task_name
                + " & " + _fmt(d_val)
                + " & " + c_cells + " & " + c_mean_s + " & " + c_geo_s
                + " & " + o_cells + " & " + o_mean_s + " & " + o_geo_s
                + r" \\"
            )

        # ─ Summary: Mean row ─────────────────────────────────────────────────
        lines.append(r"\cmidrule{2-" + str(ours_end) + "}")
        dense_mean = _arith_mean_list(dense_list)

        c_group_means = [_arith_mean_list(col_vals_cola[gk]) for gk in group_keys]
        o_group_means = [_arith_mean_list(col_vals_ours[gk]) for gk in group_keys]

        # top-3 among the 10 group means
        all_means = c_group_means + o_group_means
        top3_m = _top3_indices(all_means)
        all_means_fmt = [_fmt(v) for v in all_means]
        for rank, idx in enumerate(top3_m):
            if all_means_fmt[idx] != "--":
                all_means_fmt[idx] = _ulc(all_means_fmt[idx], rank)

        c_mean_all = _arith_mean_list(c_group_means)
        o_mean_all = _arith_mean_list(o_group_means)
        c_mean_all_s = _fmt(c_mean_all)
        o_mean_all_s = _fmt(o_mean_all)
        if not math.isnan(c_mean_all) and not math.isnan(o_mean_all):
            if o_mean_all >= c_mean_all:
                o_mean_all_s = r"\textbf{" + o_mean_all_s + "}"
            else:
                c_mean_all_s = r"\textbf{" + c_mean_all_s + "}"

        c_geo_all = _geo_mean_pct_list(
            c_group_means, [dense_mean] * n_g
        )
        o_geo_all = _geo_mean_pct_list(
            o_group_means, [dense_mean] * n_g
        )
        c_geo_all_s = _fmt(c_geo_all, pct=True)
        o_geo_all_s = _fmt(o_geo_all, pct=True)
        if not math.isnan(c_geo_all) and not math.isnan(o_geo_all):
            if o_geo_all >= c_geo_all:
                o_geo_all_s = r"\textbf{" + o_geo_all_s + "}"
            else:
                c_geo_all_s = r"\textbf{" + c_geo_all_s + "}"

        lines.append(
            r" & \textbf{Mean}"
            + " & " + _fmt(dense_mean)
            + " & " + " & ".join(all_means_fmt[:n_g])
            + " & " + c_mean_all_s + " & " + c_geo_all_s
            + " & " + " & ".join(all_means_fmt[n_g:])
            + " & " + o_mean_all_s + " & " + o_geo_all_s
            + r" \\"
        )

        # ─ Summary: Geo% row ─────────────────────────────────────────────────
        c_group_geos = [
            _geo_mean_pct_list(col_vals_cola[gk], dense_list)
            for gk in group_keys
        ]
        o_group_geos = [
            _geo_mean_pct_list(col_vals_ours[gk], dense_list)
            for gk in group_keys
        ]

        all_geos = c_group_geos + o_group_geos
        top3_g = _top3_indices(all_geos)
        all_geos_fmt = [_fmt(v, pct=True) for v in all_geos]
        for rank, idx in enumerate(top3_g):
            if all_geos_fmt[idx] != "--":
                all_geos_fmt[idx] = _ulc(all_geos_fmt[idx], rank)

        c_geo_overall = _arith_mean_list(c_group_geos)
        o_geo_overall = _arith_mean_list(o_group_geos)
        c_geo_ov_s = _fmt(c_geo_overall, pct=True)
        o_geo_ov_s = _fmt(o_geo_overall, pct=True)
        if not math.isnan(c_geo_overall) and not math.isnan(o_geo_overall):
            if o_geo_overall >= c_geo_overall:
                o_geo_ov_s = r"\textbf{" + o_geo_ov_s + "}"
            else:
                c_geo_ov_s = r"\textbf{" + c_geo_ov_s + "}"

        lines.append(
            r" & \textbf{Geo\%}"
            + " & " + _fmt(100.0, pct=True)
            + " & " + " & ".join(all_geos_fmt[:n_g])
            + " & " + c_geo_ov_s + " & "
            + " & " + " & ".join(all_geos_fmt[n_g:])
            + " & " + o_geo_ov_s + " &"
            + r" \\"
        )

        if mi < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("}")  # close \resizebox

    # Caption & label
    model_str = ", ".join(_escape(m) for m in models)
    lines.append(
        r"\caption{Comparison of COLA vs.\ \texttt{"
        + _escape(pruning_type)
        + r"} calibration sampling under "
        + _escape(compression_type)
        + r" compression ("
        + model_str
        + r").}"
    )
    lines.append(
        r"\label{tab:"
        + compression_type
        + "_"
        + pruning_type
        + "}"
    )
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX comparison table: COLA vs. a selected "
                    "calibration-sampling strategy."
    )
    parser.add_argument(
        "--compression_type",
        required=True,
        choices=["pruning", "quantization", "awq", "2ssp"],
        help="Compression method to filter on.",
        # For 2ssp, data is loaded from 2ssp_cola_experiment_results.csv
        # and 2ssp_experiment_results.csv instead of the default CSVs.
    )
    parser.add_argument(
        "--pruning_type",
        required=True,
        help="Calibration-sampling strategy from experiment_results_new.csv "
             "(e.g. words_dataset, unique_tokens, random).",
    )
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        help="One or more HuggingFace model identifiers.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for output .tex file (default: results/tables/).",
    )
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    # if args.compression_type == "2ssp":
    #     cola_df = pd.read_csv(COLA_2SSP_CSV)
    #     exp_df  = pd.read_csv(EXP_2SSP_CSV)
    # else:
    cola_df = pd.read_csv(COLA_CSV)
    exp_df  = pd.read_csv(EXP_CSV)

    for m in args.model:
        if m not in exp_df["model"].unique():
            print(f"[WARN] Model '{m}' not found in {EXP_CSV.name}", file=sys.stderr)
        if m not in cola_df["model"].unique():
            print(f"[WARN] Model '{m}' not found in {COLA_CSV.name}", file=sys.stderr)

    # ── generate ──────────────────────────────────────────────────────────────
    tex = _build_latex(args.model, args.compression_type, args.pruning_type, cola_df, exp_df)

    # ── write ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = "_".join(m.replace("/", "_") for m in args.model)
    stem = f"{safe}_{args.compression_type}_{args.pruning_type}"
    tex_path = out_dir / f"{stem}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"LaTeX -> {tex_path}")

    print("\n" + "=" * 80)
    print(tex)


if __name__ == "__main__":
    main()

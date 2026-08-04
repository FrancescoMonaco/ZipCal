import pandas as pd
import os

# Define groups from data.py
groups = {
    "General": ["c4", "oscar", "redpajama", "pile"],
    "Arithmetic Reasoning": ["gsm8k", "svamp", "mawps"],
    "NLU Inference": ["anli_r1", "esnli", "rte"],
    "Commonsense QA": ["boolq", "commonsense_qa", "race", "winogrande"],
    "Translation": ["wmt14", "iwslt"],
    "Coding": ["opc", "ds1000", "mbpp"],
}


def get_group(dataset_name):
    if not isinstance(dataset_name, str):
        return "Other"
    if "_" in dataset_name:
        return "Mixed"
    for group_name, datasets in groups.items():
        if dataset_name in datasets:
            return group_name
    return "Other"


def format_latex_comparison(df):
    # Calculate Mean row
    mean_row = df.mean(axis=0, skipna=True)

    header = " & " + " & ".join(df.columns) + " \\\\\n\\midrule\n"
    rows = []
    for task, row in df.iterrows():
        row_str = f"{task}"
        # Compare Unique Tokens vs COLA
        methods = [c for c in df.columns if c in ["unique_tokens", "cola"]]
        best_method = None
        if len(methods) == 2:
            if row["unique_tokens"] > row["cola"]:
                best_method = "unique_tokens"
            elif row["cola"] > row["unique_tokens"]:
                best_method = "cola"

        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_str += " & nan"
            elif col == best_method and best_method is not None:
                row_str += f" & \\cellcolor{{blue!15}} \\textbf{{{val:.4f}}}"
            else:
                row_str += f" & {val:.4f}"
        row_str += " \\\\"
        rows.append(row_str)

    # Add Mean Row
    mean_str = "Mean"
    for col in df.columns:
        val = mean_row[col]
        if pd.isna(val):
             mean_str += " & nan"
        else:
             mean_str += f" & {val:.4f}"
    mean_str += " \\\\"
    rows.append(mean_str)

    return (
        "\\begin{tabular}{l"
        + "r" * len(df.columns)
        + "}\n\\toprule\n"
        + header
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}"
    )


def reorganize_results(csv_path, cola_csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df_main = pd.read_csv(csv_path)

    # Ensure numeric columns are actually numeric
    cols_to_numeric = ["pruned_value", "original_value"]
    for col in cols_to_numeric:
        if col in df_main.columns:
             df_main[col] = pd.to_numeric(df_main[col], errors="coerce")

    df_main["group"] = df_main["calibration_datasets"].apply(get_group)

    df_cola = pd.DataFrame()
    if os.path.exists(cola_csv_path):
        try:
            df_cola = pd.read_csv(cola_csv_path)
            # Check if headers look like data (missing expected columns)
            # Observed format: task, metric, value, model, compression_type, sampling, nsamples, sparsity, datasets
            if "task" not in df_cola.columns and "model" not in df_cola.columns:
                 df_cola = pd.read_csv(cola_csv_path, header=None, names=["task", "metric", "pruned_value", "model", "compression_type", "sampling", "nsamples", "sparsity", "datasets"])

            # Standardize COLA columns to match main DF
            if "value" in df_cola.columns:
                df_cola = df_cola.rename(columns={"value": "pruned_value"})
            
            # Ensure pruned_value is numeric
            if "pruned_value" in df_cola.columns:
                 df_cola["pruned_value"] = pd.to_numeric(df_cola["pruned_value"], errors="coerce")

            if "datasets" in df_cola.columns:
                df_cola["group"] = df_cola["datasets"].apply(get_group)
        except Exception as e:
            print(f"Warning: Could not read COLA CSV: {e}")

    # Determine method column (pruning_type vs sampling)
    # The new file has pruning_type, old had sampling
    method_col = "pruning_type" if "pruning_type" in df_main.columns else "sampling"
    
    # Check if compression_type exists
    has_compression_type = "compression_type" in df_main.columns
    
    # Get unique compression types
    comp_types = df_main["compression_type"].unique() if has_compression_type else ["pruning"]

    for comp_type in comp_types:
        if has_compression_type:
            df_comp = df_main[df_main["compression_type"] == comp_type]
        else:
            df_comp = df_main
            
        # Filter out rows where pruned_value is NaN
        df = df_comp.dropna(subset=["pruned_value"])

        if df.empty:
            print(f"No data for compression_type={comp_type}")
            continue

        models = df["model"].unique()
        nsamples_list = df["nsamples"].unique()

        # Update output directory to include compression type if available
        if has_compression_type:
             output_dir = os.path.join("results", "reorganized", str(comp_type))
        else:
             output_dir = "results/reorganized"
             
        os.makedirs(output_dir, exist_ok=True)

        for model in models:
            model_name_safe = model.replace("/", "-")
            for nsamples in nsamples_list:
                model_ns_df = df[(df["model"] == model) & (df["nsamples"] == nsamples)]
                if model_ns_df.empty:
                    continue

                # Average across calibration groups keeping tasks and method separate
                # Check columns existence
                if "task" not in model_ns_df.columns or method_col not in model_ns_df.columns or "group" not in model_ns_df.columns:
                    print(f"Missing columns for grouping in {comp_type}, model {model}")
                    continue

                group_avg = (
                    model_ns_df.groupby(["task", method_col, "group"])["pruned_value"]
                    .mean()
                    .reset_index()
                )
                final_avg = (
                    group_avg.groupby(["task", method_col])["pruned_value"].mean().unstack()
                )
                orig_values = model_ns_df.groupby("task")["original_value"].first()
                final_avg.insert(0, "original", orig_values)

                # --- COLA vs Unique Tokens Comparison ---
                comp_data = []
                # Check for unique_tokens
                if "unique_tokens" in final_avg.columns:
                    for task, val in final_avg["unique_tokens"].items():
                        if pd.notna(val):
                            comp_data.append(
                                {"task": task, "method": "unique_tokens", "value": val}
                            )

                if not df_cola.empty:
                    cola_method_col = "sampling" if "sampling" in df_cola.columns else "pruning_type"
                    if cola_method_col in df_cola.columns:
                        cola_match = df_cola[
                            (df_cola["model"] == model)
                            & (df_cola["nsamples"] == nsamples)
                            & (df_cola[cola_method_col] == "cola")
                        ]
                        if not cola_match.empty:
                            # Use group-averaging for COLA as well for consistency if group col exists
                            if "group" in cola_match.columns:
                                cola_group_avg = (
                                    cola_match.groupby(["task", "group"])["pruned_value"]
                                    .mean()
                                    .reset_index()
                                )
                                cola_avg = (
                                    cola_group_avg.groupby("task")["pruned_value"].mean()
                                )
                            else:
                                cola_avg = cola_match.groupby("task")["pruned_value"].mean()

                            for task, val in cola_avg.items():
                                comp_data.append({"task": task, "method": "cola", "value": val})

                comparison_df = pd.DataFrame()
                if comp_data:
                    comp_df_raw = pd.DataFrame(comp_data)
                    comparison_df = comp_df_raw.pivot(
                        index="task", columns="method", values="value"
                    )
                    comparison_df.insert(0, "original", orig_values)
                    comparison_df = comparison_df.dropna()


                # Prepare Markdown version with bolding for top 3
                md_avg = final_avg.copy().astype(str)
                for task, row in final_avg.iterrows():
                    cols_to_check = [c for c in final_avg.columns if c != "original"]
                    vals = row[cols_to_check].dropna()
                    if not vals.empty:
                        top3_count = min(3, len(vals))
                        top3 = vals.nlargest(top3_count).index.tolist()
                        for col in final_avg.columns:
                            val = row[col]
                            if pd.isna(val):
                                md_avg.at[task, col] = "nan"
                            elif col in top3:
                                md_avg.at[task, col] = f"**{val:.4f}**"
                            else:
                                md_avg.at[task, col] = f"{val:.4f}"
                    else:
                        for col in final_avg.columns:
                            val = row[col]
                            if pd.isna(val):
                                md_avg.at[task, col] = "nan"
                            else:
                                md_avg.at[task, col] = f"{val:.4f}"


                # Calcola la media per ogni colonna e aggiungi la riga 'Mean'
                mean_row = final_avg.mean(axis=0, skipna=True)
                mean_row_str = {}
                for col in final_avg.columns:
                    val = mean_row[col]
                    if pd.isna(val):
                        mean_row_str[col] = "nan"
                    else:
                        mean_row_str[col] = f"{val:.4f}"
                md_avg.loc["Mean"] = mean_row_str

                # Prepare LaTeX version with highlighting

                def format_latex_table(df):
                    header = " & " + " & ".join(df.columns) + " \\\\\n\\midrule\n"
                    rows = []
                    for task, row in df.iterrows():
                        cols_to_check = [c for c in df.columns if c != "original"]
                        vals = row[cols_to_check].dropna()
                        top3_indices = []
                        if not vals.empty:
                            top3 = vals.nlargest(3)
                            top3_indices = top3.index.tolist()

                        row_str = f"{task}"
                        for col in df.columns:
                            val = row[col]
                            if pd.isna(val):
                                row_str += " & nan"
                            elif col in top3_indices:
                                rank = top3_indices.index(col)
                                color = ["green!40", "green!25", "green!10"][rank]
                                row_str += f" & \\cellcolor{{{color}}} {val:.4f}"
                            else:
                                row_str += f" & {val:.4f}"
                        row_str += " \\\\"
                        rows.append(row_str)

                    # Aggiungi la riga della media
                    mean_row = df.mean(axis=0, skipna=True)
                    mean_str = "Mean"
                    for col in df.columns:
                        val = mean_row[col]
                        if pd.isna(val):
                            mean_str += " & nan"
                        else:
                            mean_str += f" & {val:.4f}"
                    mean_str += " \\\\"
                    rows.append(mean_str)

                    return (
                        "\\begin{tabular}{l"
                        + "r" * len(df.columns)
                        + "}\n\\toprule\n"
                        + header
                        + "\n".join(rows)
                        + "\n\\bottomrule\n\\end{tabular}"
                    )

                latex_table_str = format_latex_table(final_avg)

                # Save to Markdown
                filename = f"{model_name_safe}_ns{nsamples}.md"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, "w") as f:
                    f.write(f"# Results for {model} (nsamples={nsamples}) - {comp_type}\n\n")
                    f.write("## Average across Calibration Groups\n\n")
                    f.write(md_avg.to_markdown())
                    f.write("\n\n")

                    f.write("## LaTeX Table\n\n")
                    f.write(
                        "Note: Requires `\\usepackage[table]{xcolor}` in your LaTeX preamble.\n\n"
                    )
                    f.write("```latex\n")
                    f.write(latex_table_str)
                    f.write("\n```\n\n")

                    if not comparison_df.empty:
                        f.write("## Comparison: Unique Tokens vs COLA\n\n")
                        # Add Mean row for Markdown
                        comp_df_md = comparison_df.copy()
                        mean_row = comp_df_md.mean(axis=0, skipna=True)
                        comp_df_md.loc["Mean"] = mean_row
                        f.write(comp_df_md.to_markdown())
                        f.write("\n\n")
                        f.write("### LaTeX Comparison Table\n\n")
                        f.write("```latex\n")
                        f.write(format_latex_comparison(comparison_df))
                        f.write("\n```\n")

                print(f"Saved summary for {model} ns={nsamples} ({comp_type}) to {filepath}")


if __name__ == "__main__":
    reorganize_results(
        #"results/multidataset_results.csv",
        #"results/cola_experiment_results_multidataset.csv",
        "results/experiment_results_new.csv",
        "results/cola_results_new.csv",
    )

import argparse
import os
import sys
import torch
import pandas as pd
import logging
from filelock import FileLock
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot
from sentence_transformers import SentenceTransformer

# Add 2SSP to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "2SSP"))
from src.pruning import two_stage_2ssp

from data import get_dataset, get_text_from_item
from similarity_check import prepare_calibration
from eval import evaluate_model

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)


def get_completed_experiments(output_csv, eval_tasks):
    """Returns a set of (model, compression_type, pruning_type, nsamples, sparsity, calibration_datasets)
    where all requested eval tasks are already present in the CSV.

    Note: For task groups (e.g., mmlu, cmmlu), results may be expanded into
    subtask names like "mmlu_*". We treat a task as covered if any row has an
    exact match or a prefix match (task + "_").
    """
    if not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv)
        required_cols = {"model", "compression_type", "pruning_type", "nsamples", "sparsity", "calibration_datasets"}
        if not required_cols.issubset(df.columns):
            return set()
        def _task_covered(available_tasks, requested_task):
            if requested_task in available_tasks:
                return True
            prefix = f"{requested_task}_"
            return any(t.startswith(prefix) for t in available_tasks)

        completed = set()
        grouped = (
            df.groupby(
                [
                    "model",
                    "compression_type",
                    "pruning_type",
                    "nsamples",
                    "sparsity",
                    "calibration_datasets",
                ]
            )["task"]
            .apply(set)
            .reset_index()
        )

        for _, row in grouped.iterrows():
            tasks_in_csv = row["task"]
            if all(_task_covered(tasks_in_csv, t) for t in eval_tasks):
                completed.add(
                    (
                        row["model"],
                        row["compression_type"],
                        row["pruning_type"],
                        row["nsamples"],
                        row["sparsity"],
                        row["calibration_datasets"],
                    )
                )
        return completed
    except Exception as e:
        log.error(f"Error reading completed experiments: {e}")
        return set()


def get_existing_original_results(output_csv, model_name):
    """Search for cached original (dense) model results across all CSVs in the
    results/ directory, not just the current output CSV.  This avoids
    re-evaluating the dense model when the same model was already evaluated in
    a different experiment run."""
    results_map = {}  # (task, metric) -> value  — deduplicates automatically

    # Gather candidate CSV files: the target CSV + every CSV under results/ and
    # the output CSV's directory (recursively), to maximize cache hits.
    csv_candidates = set()
    if os.path.exists(output_csv):
        csv_candidates.add(output_csv)

    search_roots = set()
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        search_roots.add(output_dir)
    search_roots.add("results")

    for root in list(search_roots):
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.endswith(".csv"):
                    csv_candidates.add(os.path.join(dirpath, fname))

    for csv_path in csv_candidates:
        try:
            df = pd.read_csv(csv_path)
            if "original_value" not in df.columns or "model" not in df.columns:
                continue
            model_df = df[df["model"] == model_name]
            if model_df.empty:
                continue
            subset = model_df[["task", "metric", "original_value"]].drop_duplicates()
            subset = subset[subset["original_value"].notnull()]
            for _, row in subset.iterrows():
                key = (row["task"], row["metric"])
                if key not in results_map:
                    results_map[key] = row["original_value"]
        except Exception:
            continue

    return [
        {"task": t, "metric": m, "value": v}
        for (t, m), v in results_map.items()
    ]


def _task_covered(available_tasks, requested_task):
    """Return True if requested_task is present or covered by a group prefix.

    For group tasks like mmlu/cmmlu, results are recorded as subtask names
    (e.g., mmlu_high_school_physics). We treat those as covering the group.
    """
    if requested_task in available_tasks:
        return True
    prefix = f"{requested_task}_"
    return any(t.startswith(prefix) for t in available_tasks)


def _task_in_requests(existing_task, requested_tasks):
    """Return True if an existing task belongs to the requested tasks list.

    This allows group tasks (e.g., requested "mmlu") to match "mmlu_*".
    """
    for req in requested_tasks:
        if existing_task == req or existing_task.startswith(f"{req}_"):
            return True
    return False


def get_tokenized_data(dataset, tokenizer, dataset_name, max_length=128):
    # Collect all texts first, then batch-tokenize for speed
    texts = [get_text_from_item(item, dataset_name) for item in dataset]
    processed_dataset = []
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        for j in range(len(batch_texts)):
            processed_dataset.append(
                {
                    "input_ids": encoded["input_ids"][j],
                    "attention_mask": encoded["attention_mask"][j],
                }
            )
    return processed_dataset


def process_results(results_dict):
    """Extracts clean metrics from lm_eval output."""
    processed = []
    if "results" in results_dict:
        for task, metrics in results_dict["results"].items():
            for metric_name, value in metrics.items():
                # Filter out stderr and non-numeric values
                if isinstance(value, (int, float)) and "stderr" not in metric_name:
                    processed.append(
                        {"task": task, "metric": metric_name, "value": value}
                    )
    return processed


def main():
    parser = argparse.ArgumentParser(description="Run Pruning Experiment with lm_eval")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["winogrande"], help="Datasets for calibration"
    )
    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        default=[
            "boolq",
            "rte",
            "hellaswag",
            "winogrande",
            "arc_challenge",
            "arc_easy",
            "openbookqa",
            "anli_r1",
            "gsm8k",
            "mmlu",
        ],
        help="Tasks for evaluation (lm_eval names)",
    )
    parser.add_argument(
        "--compression_type",
        type=str,
        choices=["pruning", "quantization", "awq", "2ssp"],
        default="pruning",
        help="Type of compression to perform",
    )
    parser.add_argument(
        "--pruning_types",
        nargs="+",
        choices=[
            "most_similar",
            "random",
            "decoupled",
            "most_dissimilar",
            "least_perplexity",
            "herding",
            "distribution_matching",
            "distribution_matching_no_outliers",
            "zipf",
            "shuffled_zipf",
            "unique_tokens",
            "random_words",
            "words_dataset",
            "dictionary"
        ],
        default=["random_words"],
        help="Types of pruning to perform",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples"
    )
    parser.add_argument("--sparsity", type=float, default=0.5, help="Pruning sparsity")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/experiment_results_new.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--save_models", action="store_true", help="Save pruned models to disk"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/pruned",
        help="Directory to save pruned models",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 0. Early exit if all requested experiments already exist in the CSV
    calib_name = "_".join(args.datasets)
    completed = get_completed_experiments(args.output_csv, args.eval_tasks)
    remaining_types = [
        t for t in args.pruning_types
        if (args.model, args.compression_type, t, args.nsamples, args.sparsity, calib_name) not in completed
    ]
    if not remaining_types:
        log.info("All requested experiments are already in the CSV. Nothing to do.")
        return
    log.info(f"Experiments to run: {remaining_types} (skipping {len(args.pruning_types) - len(remaining_types)} already completed)")

    # 1. Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Evaluate original model once
    existing_orig_metrics = get_existing_original_results(args.output_csv, args.model)
    existing_tasks = set(m["task"] for m in existing_orig_metrics)
    tasks_to_eval = [
        t for t in args.eval_tasks if not _task_covered(existing_tasks, t)
    ]

    model = None
    if tasks_to_eval:
        log.info(
            f"Loading original model {args.model} for initial evaluation of tasks: {tasks_to_eval}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        log.info(f"Evaluating original model on: {tasks_to_eval}")
        orig_raw = evaluate_model(args.model, model, tokenizer, tasks_to_eval)
        new_orig_metrics = process_results(orig_raw)
        orig_metrics = [
            m for m in existing_orig_metrics if _task_in_requests(m["task"], args.eval_tasks)
        ] + new_orig_metrics
    else:
        log.info(
            f"All tasks {args.eval_tasks} already have original results. Skipping initial evaluation."
        )
        orig_metrics = [
            m for m in existing_orig_metrics if _task_in_requests(m["task"], args.eval_tasks)
        ]

    # Prepare calibration data base (tokenized) once
    log.info("Preparing base tokenized data for calibration...")
    all_tokenized_datasets = []  # List of lists, one per dataset (for coreset resampling)
    for d_name in args.datasets:
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            continue
        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            dataset = (
                raw_dataset.get("train")
                or raw_dataset.get("test")
                or raw_dataset[list(raw_dataset.keys())[0]]
            )
        else:
            dataset = raw_dataset
        tokenized_data = get_tokenized_data(dataset, tokenizer, d_name)
        all_tokenized_datasets.append(tokenized_data)

    calibration_type_map = {
        "most_similar": "prototype",
        "most_dissimilar": "most_different",
        "decoupled": "decoupled",
        "least_perplexity": "least_perplexity",
        "random": "random_sample",
        "herding": "herding",
        "distribution_matching": "distribution_matching",
        "distribution_matching_no_outliers": "distribution_matching_no_outliers",
        "zipf": "zipf",
        "shuffled_zipf": "shuffled_zipf",
        "unique_tokens": "unique_tokens",
        "random_words": "random_words",
        "words_dataset": "words_dataset",
        "dictionary": "dictionary",
    }

    # 3. Loop through pruning types
    safe_model_name = args.model.replace("/", "-")

    sentence_transformer = None  # Lazy-loaded once, reused across pruning types
    for p_type in args.pruning_types:
        exp_key = (args.model, args.compression_type, p_type, args.nsamples, args.sparsity, calib_name)
        if exp_key in completed:
            log.info(f"Skipping {p_type} (already in CSV for model={args.model}, compression={args.compression_type}, nsamples={args.nsamples}, sparsity={args.sparsity}, datasets={calib_name})")
            continue

        log.info(
            f"\n--- Starting process for {args.compression_type} type: {p_type} ---"
        )

        # Define save path
        save_path = os.path.join(
            args.models_dir,
            safe_model_name,
            args.compression_type,
            p_type,
            str(args.nsamples),
            str(args.sparsity),
            calib_name,
        )

        pruned_model = None
        if os.path.exists(os.path.join(save_path, "config.json")):
            log.info(f"Found existing compressed model at {save_path}. Loading...")
            pruned_model = AutoModelForCausalLM.from_pretrained(
                save_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            log.info(f"No existing model found at {save_path}. Compressing...")

            if model is None:
                log.info(
                    f"Loading original model {args.model} for {args.compression_type}..."
                )
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            # Prepare calibration for this specific type
            calib_model = model if p_type == "least_perplexity" else None
            if calib_model is None:
                if sentence_transformer is None:
                    sentence_transformer = SentenceTransformer("all-MiniLM-L12-v2", device=device)
                calib_model = sentence_transformer

            calibration_data_dicts = prepare_calibration(
                model=calib_model,
                dataloader=all_tokenized_datasets,
                nsamples=args.nsamples,
                type=calibration_type_map[p_type],
                distance="flatten",
                model_name=safe_model_name,
                dataset_name=calib_name,
                tokenizer=tokenizer,
            )

            # Compress
            if args.compression_type == "2ssp":
                log.info(f"Structured pruning with 2SSP, sparsity {args.sparsity}...")
                # Convert calibration data to 2SSP format: list of (1, seq_len) tensors
                all_ids = torch.cat([item["input_ids"].view(-1) for item in calibration_data_dicts])
                ssp_seq_len = 2048
                num_chunks = all_ids.size(0) // ssp_seq_len
                if num_chunks == 0:
                    log.warning(f"Only {all_ids.size(0)} tokens, need >= {ssp_seq_len} for 2SSP. Using all tokens as one sample.")
                    calibration_2ssp = [all_ids.unsqueeze(0)]
                else:
                    calibration_2ssp = [all_ids[i * ssp_seq_len : (i + 1) * ssp_seq_len].unsqueeze(0) for i in range(num_chunks)]
                log.info(f"2SSP calibration: {len(calibration_2ssp)} samples of length {calibration_2ssp[0].size(1)}")
                model.config.use_cache = False
                result = two_stage_2ssp(model, calibration_2ssp, args.sparsity)
                if result is False:
                    log.error("2SSP pruning failed – invalid sparsity parameters")
                    continue
                model = result
            else:
                data_list = [
                    {
                        "input_ids": item["input_ids"].cpu().tolist(),
                        "attention_mask": item["attention_mask"].cpu().tolist(),
                    }
                    for item in calibration_data_dicts
                ]
                calibration_dataset = Dataset.from_list(data_list)

                if args.compression_type == "pruning":
                    log.info(f"Pruning model with sparsity {args.sparsity}...")
                    recipe = WandaPruningModifier(
                        sparsity=args.sparsity, mask_structure="0:0", targets="__ALL__"
                    )
                elif args.compression_type == "quantization":
                    log.info("Quantizing model with GPTQ...")
                    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
                elif args.compression_type == "awq":
                    log.info("Quantizing model with AWQ...")
                    recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

                oneshot(model=model, dataset=calibration_dataset, recipe=recipe)
            log.info("Compression complete.")

            # if args.save_models:
            #     log.info(f"Saving pruned model to {save_path}...")
            #     os.makedirs(save_path, exist_ok=True)
            #     model.save_pretrained(save_path)
            #     tokenizer.save_pretrained(save_path)

            pruned_model = model

        # 4. Evaluate pruned model
        log.info(f"Evaluating compressed model ({p_type})...")
        pruned_raw = evaluate_model(
            args.model, pruned_model, tokenizer, args.eval_tasks
        )
        pruned_metrics = process_results(pruned_raw)

        # 5. Save results to CSV
        log.info(f"Saving results for {p_type} to {args.output_csv}...")
        rows = []
        for orig in orig_metrics:
            pruned_val = next(
                (
                    p["value"]
                    for p in pruned_metrics
                    if p["task"] == orig["task"] and p["metric"] == orig["metric"]
                ),
                None,
            )
            rows.append(
                {
                    "model": args.model,
                    "task": orig["task"],
                    "metric": orig["metric"],
                    "original_value": orig["value"],
                    "pruned_value": pruned_val,
                    "compression_type": args.compression_type,
                    "pruning_type": p_type,
                    "nsamples": args.nsamples,
                    "sparsity": args.sparsity,
                    "calibration_datasets": calib_name,
                }
            )

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

        # Use a lock file to prevent race conditions when multiple scripts write to the same CSV
        lock_path = args.output_csv + ".lock"
        lock = FileLock(lock_path)

        with lock:
            df.to_csv(
                args.output_csv,
                mode="a",
                header=not os.path.exists(args.output_csv),
                index=False,
            )

        # If we are going to the next p_type, we MUST reload the original model
        # because 'model' (which is 'pruned_model') is now modified.
        if p_type != remaining_types[-1]:
            log.info("Reloading original model for the next pruning type...")
            if model is not None:
                del model
            del pruned_model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

    log.info("All experiments finished successfully.")


if __name__ == "__main__":
    main()

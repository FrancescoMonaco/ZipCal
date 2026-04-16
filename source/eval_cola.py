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

# Add COLA to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "COLA"))
from cola.sample_selection import select_samples

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


def get_tokenized_data(dataset, tokenizer, dataset_name, max_length=128):
    processed_dataset = []
    for item in dataset:
        text = get_text_from_item(item, dataset_name)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        processed_dataset.append(
            {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
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


def get_completed_experiments(output_csv):
    """Returns a set of (model, pruning_type, sampling, nsamples, sparsity, datasets)
    for which at least one result row already exists in the CSV."""
    if not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv)
        required_cols = {"model", "pruning_type", "sampling", "nsamples", "sparsity", "datasets"}
        if not required_cols.issubset(df.columns):
            return set()
        return set(
            zip(
                df["model"],
                df["pruning_type"],
                df["sampling"],
                df["nsamples"],
                df["sparsity"],
                df["datasets"],
            )
        )
    except Exception as e:
        log.error(f"Error reading completed experiments: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Run Pruning Experiment with COLA and other methods"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B", help="Model name or path"
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
        default="pruning",
        choices=["pruning", "quantization", "awq", "2ssp"],
        help="Type of compression to perform",
    )
    parser.add_argument(
        "--pruning_types",
        nargs="+",
        choices=[
            "cola",
            "most_similar",
            "random",
            "decoupled",
            "most_dissimilar",
            "least_perplexity",
            "herding",
            "distribution_matching",
            "distribution_matching_no_outliers",
            "zipf",
            "unique_tokens",
        ],
        default=["cola"],
        help="Types of pruning to perform",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples"
    )
    parser.add_argument("--sparsity", type=float, default=0.5, help="Pruning sparsity")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/cola_experiment_results.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="Maximum sequence length for calibration",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=4080,
        help="Maximum number of candidate samples for COLA",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 0. Early exit if all requested experiments already exist in the CSV
    datasets_key = ",".join(args.datasets)
    completed = get_completed_experiments(args.output_csv)
    remaining_types = [
        t for t in args.pruning_types
        if (args.model, args.compression_type, t, args.nsamples, args.sparsity, datasets_key) not in completed
    ]
    if not remaining_types:
        log.info("All requested experiments are already in the CSV. Nothing to do.")
        return
    log.info(f"Experiments to run: {remaining_types} (skipping {len(args.pruning_types) - len(remaining_types)} already completed)")
    args.pruning_types = remaining_types

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare calibration data base
    log.info("Preparing base data for calibration...")
    all_raw_texts = []
    all_tokenized_data = []

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

        # For COLA we need raw texts
        for i, item in enumerate(dataset):
            if i >= args.max_candidates:
                break
            text = get_text_from_item(item, d_name)
            all_raw_texts.append(text)

        # For other methods we need tokenized data
        tokenized_data = get_tokenized_data(
            dataset, tokenizer, d_name, max_length=args.max_seq_len
        )
        all_tokenized_data.extend(tokenized_data)

    calibration_type_map = {
        "most_similar": "prototype",
        "most_dissimilar": "most_different",
        "decoupled": "decoupled",
        "least_perplexity": "perplexity",
        "herding": "herding",
        "distribution_matching": "distribution_matching",
        "distribution_matching_no_outliers": "distribution_matching_no_outliers",
        "zipf": "zipf",
        "unique_tokens": "unique_tokens",
    }

    results_list = []

    for p_type in args.pruning_types:

        log.info(f"\n--- Starting Pruning Type: {p_type} ---")

        # Load model for pruning
        log.info(f"Loading model {args.model} for pruning...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 3. Prepare calibration data for this pruning type
        log.info(f"Preparing calibration data for {p_type}...")
        if p_type == "cola":
            # Limit candidates to avoid OOM and long processing
            candidates = all_raw_texts[: args.max_candidates]
            processed_samples = [{"text": t} for t in candidates]

            # COLA select_samples expects a model and tokenizer
            selected_cola_samples = select_samples(
                processed_samples,
                model,
                tokenizer,
                num_clusters=args.nsamples,
                device=device,
                batch_size=16,
            )

            if not selected_cola_samples:
                raise ValueError(
                    "COLA failed to select samples. Check if transformer layers were detected correctly."
                )

            # Tokenize selected samples for llmcompressor
            calibration_data = []
            for sample in selected_cola_samples:
                encoded = tokenizer(
                    sample["text"],
                    truncation=True,
                    max_length=args.max_seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                calibration_data.append(
                    {
                        "input_ids": encoded["input_ids"].squeeze(0),
                        "attention_mask": encoded["attention_mask"].squeeze(0),
                    }
                )
        elif p_type == "random":
            import random

            indices = random.sample(
                range(len(all_tokenized_data)),
                min(args.nsamples, len(all_tokenized_data)),
            )
            calibration_data = [all_tokenized_data[i] for i in indices]
        else:
            calibration_data = prepare_calibration(
                all_tokenized_data,
                args.nsamples,
                method=calibration_type_map[p_type],
                model_name=args.model,
            )

        # 4. Prune or Quantize the model
        log.info(f"Compressing model with {args.compression_type} using {p_type} calibration data...")

        if args.compression_type == "2ssp":
            # Convert calibration data to 2SSP format: list of (1, seq_len) tensors
            all_ids = torch.cat([item["input_ids"].view(-1) for item in calibration_data])
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
                del model
                torch.cuda.empty_cache()
                continue
            model = result
        else:
            # Convert calibration data to Dataset object for llmcompressor
            calibration_dataset = Dataset.from_list(calibration_data)

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
                recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
            oneshot(model=model, dataset=calibration_dataset, recipe=recipe)

        # 5. Evaluate pruned model
        log.info(f"Evaluating pruned model ({p_type})...")
        pruned_raw = evaluate_model(
            f"{args.model}_pruned_{p_type}", model, tokenizer, args.eval_tasks
        )
        pruned_metrics = process_results(pruned_raw)

        for m in pruned_metrics:
            m.update(
                {
                    "model": args.model,
                    "pruning_type": args.compression_type,
                    "sampling": p_type,
                    "nsamples": args.nsamples,
                    "sparsity": args.sparsity,
                    "datasets": ",".join(args.datasets),
                }
            )
            results_list.append(m)

        # Clean up to free memory
        del model
        torch.cuda.empty_cache()

    # 6. Save results
    if results_list:
        df = pd.DataFrame(results_list)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

        # Use a lock file to prevent race conditions when multiple scripts write to the same CSV
        lock_path = args.output_csv + ".lock"
        lock = FileLock(lock_path)

        with lock:
            # If file exists, append without header
            if os.path.exists(args.output_csv):
                df.to_csv(args.output_csv, mode="a", header=False, index=False)
            else:
                df.to_csv(args.output_csv, index=False)

        log.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()

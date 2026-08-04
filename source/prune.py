from similarity_check import (
    prepare_calibration,
)
from wanda_analysis import WandaAnalysis
from eval import evaluate_model
import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from sentence_transformers import SentenceTransformer
from data import get_dataset, get_text_from_item
from datasets import Dataset

from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot
import logging
from prettytable import PrettyTable
import argparse

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)


# --- MODIFIED: get_tokenized_data ---
# This function is now used for *both* similarity sampling and evaluation.
# 1. It no longer returns PyTorch tensors (return_tensors="pt" removed).
# 2. It returns a dictionary with 'input_ids' and 'attention_mask' as Python lists.
# 3. It no longer does max_length padding (this is deferred to the Data Collator).
def get_tokenized_data(
    dataset, tokenizer, dataset_name, max_length=128, return_tensors=False
):
    processed_dataset = []
    for item in dataset:
        text = get_text_from_item(item, dataset_name)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            # return_tensors="pt" is now optional/controlled by argument
        )

        # NOTE: We return the raw dict if we don't need tensors yet (for DataCollator)
        if return_tensors:
            # For similarity check, we might still need tensors in the loop
            encoded = tokenizer.pad(
                encoded,
                padding="max_length",
                return_tensors="pt",
                max_length=max_length,
            )
            processed_dataset.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "text": text,
                }
            )
        else:
            # For the DataLoader, return list/arrays, let the Collator do the rest
            processed_dataset.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "text": text,
                }
            )

    return processed_dataset


if __name__ == "__main__":
    # Pruning type from command line
    parser = argparse.ArgumentParser(description="Wanda Pruning Script")
    parser.add_argument(
        "--pruning_type",
        type=str,
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
            "unique_tokens",
        ],
        default="most_similar",
        help="Type of pruning to perform: 'most_similar', 'random', 'decoupled', 'most_dissimilar', or 'least_perplexity'",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["winogrande"],
        help="List of datasets to use for calibration",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name or path to prune",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of samples to use for calibration",
    )
    parser.add_argument(
        "--run_pruning",
        action="store_true",
        help="If set, continue past analysis and run pruning/oneshot",
    )
    args = parser.parse_args()
    pruning_type = args.pruning_type
    model_name = args.model
    nsamples = args.nsamples

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Loaded model {model_name} for embedding extraction.")

    dataset_names = args.datasets
    log.info(f"Datasets to use: {dataset_names}")

    sentence_trsf = SentenceTransformer("all-MiniLM-L12-v2", device="cuda")
    # subset_size = 5000

    all_tokenized_data = []

    for d_name in dataset_names:
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            log.warning(f"Could not load dataset {d_name}, skipping.")
            continue

        # Determine split
        if isinstance(raw_dataset, dict) or hasattr(raw_dataset, "keys"):
            if "train" in raw_dataset.keys():
                dataset = raw_dataset["train"]
            elif "test" in raw_dataset.keys():
                dataset = raw_dataset["test"]
            else:
                dataset = raw_dataset[list(raw_dataset.keys())[0]]
        else:
            dataset = raw_dataset

        log.info(f"Loaded dataset {d_name} with {len(dataset)} samples.")

        # if len(dataset) > subset_size:
        #     dataset = dataset.select(range(subset_size))

        # Tokenize
        # We pass return_tensors=True to get_tokenized_data for the sampling part
        tokenized_data = get_tokenized_data(
            dataset, tokenizer, d_name, return_tensors=True
        )
        all_tokenized_data.extend(tokenized_data)

    log.info(f"Total tokenized samples: {len(all_tokenized_data)}")

    # Use a combined name for saving results
    dataset_name = "_".join(dataset_names)

    # Map pruning_type to prepare_calibration type
    calibration_type = "prototype"
    if pruning_type == "most_similar":
        calibration_type = "prototype"
    elif pruning_type == "most_dissimilar":
        calibration_type = "most_different"
    elif pruning_type == "decoupled":
        calibration_type = "decoupled"
    elif pruning_type == "least_perplexity":
        calibration_type = "least_perplexity"
    elif pruning_type == "random":
        calibration_type = "random_sample"
    elif pruning_type == "zipf":
        calibration_type = "zipf"
    elif pruning_type == "unique_tokens":
        calibration_type = "unique_tokens"

    # Pick the calibration data from the dataset
    calibration_data_dicts = prepare_calibration(
        model=model if pruning_type == "least_perplexity" else sentence_trsf,
        dataloader=[all_tokenized_data],
        nsamples=nsamples,
        type=calibration_type,
        distance="flatten",
        save_calibration_distribution=False,
        model_name=model_name.replace("/", "-"),
        dataset_name=dataset_name,
        tokenizer=tokenizer,
    )
    log.info(f"Calibration data prepared: {len(calibration_data_dicts)} samples.")

    # Convert calibration data to format expected by oneshot
    # Assuming calibration_data_dicts returns dicts like {'input_ids': tensor(seq_len), ...}
    # We need to extract the Python list/array of IDs from the tensor for Dataset.from_list
    data_list = []
    for item in calibration_data_dicts:
        input_ids = item["input_ids"]
        attention_mask = item.get("attention_mask")

        # Convert tensor to list of integers
        if isinstance(input_ids, torch.Tensor):
            # Ensure it's on CPU and convert to list
            if input_ids.dim() == 2 and input_ids.shape[0] == 1:
                input_ids = input_ids.squeeze(0)
            input_ids = input_ids.cpu().numpy().tolist()

            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                if attention_mask.dim() == 2 and attention_mask.shape[0] == 1:
                    attention_mask = attention_mask.squeeze(0)
                attention_mask = attention_mask.cpu().numpy().tolist()

        data_dict = {"input_ids": input_ids}
        if attention_mask is not None:
            data_dict["attention_mask"] = attention_mask

        data_list.append(data_dict)

    calibration_dataset = Dataset.from_list(data_list)

    # Collect Wanda statistics before pruning
    wanda_analyzer = WandaAnalysis(model, pruning_type=pruning_type)
    wanda_analyzer.collect(
        DataLoader(
            calibration_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, max_length=128),
        )
    )
    wanda_analyzer.compute_scores()
    wanda_analyzer.compute_activations_stats()

    import os

    save_dir = os.path.join(
        "results", model_name.replace("/", "-"), pruning_type, str(nsamples)
    )
    save_path = os.path.join(save_dir, f"{dataset_name}.pdf")

    wanda_analyzer.plot(save_path=save_path)
    wanda_analyzer.remove_hooks()
    # If not explicitly requested, stop after analysis/plots
    if not args.run_pruning:
        # TODO continue and save the pruned model's weights when --run_pruning is set
        exit(0)
    # Define Wanda recipe
    recipe = WandaPruningModifier(sparsity=0.5, mask_structure="0:0", targets="__ALL__")

    log.info("Starting Wanda pruning...")
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        # max_seq_length=128,
        # num_calibration_samples=len(calibration_data)
    )

    log.info("Pruning finished.")

    # Evaluate the pruned model
    log.info("Evaluating the pruned model...")

    results = evaluate_model(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        task_list=dataset_names,
    )
    log.info("Pruned model evaluation completed.")

    # Summarize results in a table
    table = PrettyTable()
    table.field_names = ["Task", "Metric", "Value"]

    if results is not None and "results" in results:
        for task, metrics in results["results"].items():
            for metric, value in metrics.items():
                # We only care about the main metric values, not the stderr or other metadata for now
                # unless the user wants them. Let's print float values which are likely the metrics.
                if isinstance(value, (int, float)) and "stderr" not in metric:
                    table.add_row([task, metric, f"{value:.4f}"])

    print(table)

    # Send wandb info online
    # wandb.finish()

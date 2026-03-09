import argparse
import os
import torch
import pandas as pd
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot
import sys

# Aggiungi COLA e source al sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "COLA"))

from data import get_dataset, get_text_from_item
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

def load_raw_texts(dataset_name: str, max_count: int = 4000) -> list[str]:
    """Load raw texts from a dataset. C4 is loaded via streaming to avoid full download."""
    from datasets import load_dataset

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = []
        for item in ds:
            text = item.get("text", "")
            if text and len(text.strip()) > 0:
                texts.append(text)
            if len(texts) >= max_count:
                break
        return texts

    elif dataset_name.startswith("c4"):
        # Use streaming to avoid downloading the full dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = []
        for item in ds:
            text = item.get("text", "")
            if text and len(text.strip()) > 0:
                texts.append(text)
            if len(texts) >= max_count:
                break
        return texts

    else:
        raw = get_dataset(dataset_name)
        if raw is None:
            raise ValueError(f"Could not load dataset {dataset_name}.")
        if isinstance(raw, dict):
            raw = raw.get("train") or raw.get("test") or raw[list(raw.keys())[0]]
        texts = []
        for item in raw:
            text = get_text_from_item(item, dataset_name)
            if text and len(text.strip()) > 0:
                texts.append(text)
            if len(texts) >= max_count:
                break
        return texts


def tokenize_texts(texts: list[str], tokenizer, max_seq_len: int) -> list[dict]:
    """Tokenize a list of raw texts into input_ids / attention_mask dicts."""
    tokenized = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized.append(
            {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
        )
    return tokenized


def build_recipe(exp_type: str, sparsity: float):
    """Return the llmcompressor recipe for a given compression technique."""
    if exp_type == "wanda":
        return WandaPruningModifier(sparsity=sparsity, targets="__ALL__")
    elif exp_type == "gptq":
        return GPTQModifier(targets="Linear", scheme="W4A16")
    elif exp_type == "awq":
        return AWQModifier(targets="Linear", scheme="W4A16")
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Perplexity across compression techniques")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B", help="Base model name")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.25, help="Pruning sparsity (used by Wanda)")
    parser.add_argument("--ppl_tasks", nargs="+", default=["wikitext", "c4"], help="Tasks for perplexity evaluation")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max seq len for calibration")
    parser.add_argument("--output_csv", type=str, default="results/perplexity_comparison.csv", help="Output file")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Tokenizer (shared across all runs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Define datasets to evaluate: wikitext (full) and c4 (streamed, 1100 samples)
    dataset_configs = [
        ("wikitext", 4000),
        ("c4",       1100),
    ]

    # Three compression techniques + baseline
    experiments = ["original", "wanda", "gptq", "awq"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    csv_exists = os.path.isfile(args.output_csv)

    for dataset_name, max_count in dataset_configs:
        log.info(f"\n=== Dataset: {dataset_name} (max {max_count} samples) ===")

        # Load raw texts – c4 is streamed, no full download
        log.info(f"Loading {dataset_name}...")
        all_raw_texts = load_raw_texts(dataset_name, max_count=max_count)
        log.info(f"Loaded {len(all_raw_texts)} texts from {dataset_name}")

        log.info(f"Tokenizing {len(all_raw_texts)} samples...")
        all_tokenized = tokenize_texts(all_raw_texts, tokenizer, args.max_seq_len)

        # Build calibration subset (first nsamples)
        calib_raw   = all_raw_texts[:args.nsamples]
        calib_token = all_tokenized[:args.nsamples]

        for exp_type in experiments:
            log.info(f"\n--- [{dataset_name}] Evaluating: {exp_type} ---")

            # Fresh model for every experiment
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )

            if exp_type != "original":
                log.info(f"Applying compression technique: {exp_type}...")
                calibration_dataset = Dataset.from_list(calib_token)
                recipe = build_recipe(exp_type, args.sparsity)
                oneshot(model=model, dataset=calibration_dataset, recipe=recipe)

            log.info(f"Running evaluation on tasks: {args.ppl_tasks}...")
            results = evaluate_model(
                f"{args.model}_{exp_type}_{dataset_name}",
                model,
                tokenizer,
                args.ppl_tasks,
                apply_chat_template=False,
            )

            # Collect metrics
            rows = []
            if "results" in results:
                for task, metrics in results["results"].items():
                    for m_name, val in metrics.items():
                        if isinstance(val, (int, float)) and "stderr" not in m_name:
                            rows.append({
                                "experiment":    exp_type,
                                "dataset":       dataset_name,
                                "task":          task,
                                "metric":        m_name,
                                "value":         val,
                                "sparsity":      args.sparsity if exp_type == "wanda" else 0,
                                "model":         args.model,
                                "nsamples":      args.nsamples,
                            })

            # Append to CSV (write header only if file does not exist yet)
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(
                    args.output_csv,
                    mode="a",
                    index=False,
                    header=not csv_exists,
                )
                csv_exists = True  # header already written from now on
                log.info(f"Appended {len(rows)} rows to {args.output_csv}")

            del model
            torch.cuda.empty_cache()

    log.info(f"\nAll results saved to {args.output_csv}")

if __name__ == "__main__":
    main()

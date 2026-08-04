import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from datasets import Dataset
from filelock import FileLock
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import get_dataset, get_text_from_item
from similarity_check import prepare_calibration
from llmcompressor.modifiers.pruning import WandaPruningModifier
from llmcompressor import oneshot

# Add 2SSP to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "2SSP"))
from src.pruning import two_stage_2ssp

FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)


CALIBRATION_TYPE_MAP = {
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_tokenized_data(dataset, tokenizer, dataset_name, max_length=128):
    texts = [get_text_from_item(item, dataset_name) for item in dataset]
    processed_dataset = []
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
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


def load_model(model_name):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


def build_calibration_data(args, tokenizer, device, model_name):
    all_tokenized_datasets = []
    for d_name in args.datasets:
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            log.warning(f"Could not load dataset {d_name}, skipping.")
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

    calibration_type = CALIBRATION_TYPE_MAP[args.calibration_type]

    if calibration_type == "least_perplexity":
        log.info("Using model for least_perplexity calibration selection.")
        calib_model = load_model(args.model)
    else:
        calib_model = SentenceTransformer("all-MiniLM-L12-v2", device=device)

    calibration_data_dicts = prepare_calibration(
        model=calib_model,
        dataloader=all_tokenized_datasets,
        nsamples=args.nsamples,
        type=calibration_type,
        distance="flatten",
        model_name=model_name,
        dataset_name="_".join(args.datasets),
        tokenizer=tokenizer,
    )

    if calibration_type == "least_perplexity":
        del calib_model

    return calibration_data_dicts


def build_oneshot_dataset(calibration_data_dicts):
    data_list = []
    for item in calibration_data_dicts:
        input_ids = item["input_ids"]
        attention_mask = item.get("attention_mask")
        if isinstance(input_ids, torch.Tensor):
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

    return Dataset.from_list(data_list)


def build_2ssp_calibration(calibration_data_dicts, seq_len=2048):
    all_ids = torch.cat([item["input_ids"].view(-1) for item in calibration_data_dicts])
    num_chunks = all_ids.size(0) // seq_len
    if num_chunks == 0:
        log.warning(
            f"Only {all_ids.size(0)} tokens, need >= {seq_len} for 2SSP. Using all tokens as one sample."
        )
        return [all_ids.unsqueeze(0)]
    return [
        all_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0)
        for i in range(num_chunks)
    ]


def select_matrix_names(model, num_matrices, seed):
    candidates = []
    for module_name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() < 2:
            continue
        candidates.append(module_name)

    if num_matrices is None or num_matrices <= 0:
        return None

    if len(candidates) <= num_matrices:
        return set(candidates)

    rng = random.Random(seed)
    return set(rng.sample(candidates, num_matrices))


def extract_weight_matrices(
    model,
    model_name,
    compression_name,
    sparsity,
    seed,
    selected_names,
    matrix_dir,
    index_handle,
    index_lock,
):
    rows_written = 0
    for module_name, module in model.named_modules():
        if selected_names is not None and module_name not in selected_names:
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() < 2:
            continue

        tensor = weight.detach().cpu().contiguous()
        original_dtype = str(tensor.dtype).replace("torch.", "")
        saved_tensor = tensor
        saved_dtype = original_dtype

        if tensor.dtype == torch.bfloat16:
            saved_tensor = tensor.to(torch.float16)
            saved_dtype = "float16"
        elif tensor.dtype not in (torch.float16, torch.float32):
            saved_tensor = tensor.to(torch.float16)
            saved_dtype = "float16"

        np_array = saved_tensor.numpy()
        file_name = f"{compression_name}__{module_name.replace('.', '_')}.npz"
        file_path = os.path.join(matrix_dir, file_name)
        np.savez_compressed(
            file_path,
            data=np_array,
            shape=np_array.shape,
            dtype=str(np_array.dtype),
        )

        record = {
            "model_name": model_name,
            "compression_name": compression_name,
            "layer_name": module_name,
            "matrix_name": "weight",
            "sparsity": float(sparsity),
            "seed": int(seed),
            "dtype": original_dtype,
            "saved_dtype": saved_dtype,
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "path": file_path,
        }
        with index_lock:
            index_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            index_handle.flush()
        rows_written += 1

    return rows_written


def main():
    parser = argparse.ArgumentParser(
        description="Run Wanda + 2SSP and save pruned weight matrices to Parquet."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["winogrande"], help="Calibration datasets"
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Pruning sparsity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--calibration_type",
        type=str,
        choices=list(CALIBRATION_TYPE_MAP.keys()),
        default="random_words",
        help="Calibration sampling strategy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write matrix files + NDJSON index",
    )
    parser.add_argument(
        "--num_matrices",
        type=int,
        default=None,
        help="Randomly sample N matrices per compression (default: all)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    safe_model_name = args.model.replace("/", "-")
    output_dir = args.output_dir or f"{safe_model_name}_matrices"
    os.makedirs(output_dir, exist_ok=True)
    matrix_dir = os.path.join(output_dir, "matrices")
    os.makedirs(matrix_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "index.ndjson")
    index_lock = FileLock(index_path + ".lock")

    log.info("Preparing calibration data...")
    calibration_data_dicts = build_calibration_data(
        args=args,
        tokenizer=tokenizer,
        device=device,
        model_name=safe_model_name,
    )

    log.info("Building calibration datasets for Wanda and 2SSP...")
    oneshot_dataset = build_oneshot_dataset(calibration_data_dicts)
    calibration_2ssp = build_2ssp_calibration(calibration_data_dicts)

    selected_names = None
    if args.num_matrices is not None and args.num_matrices > 0:
        log.info("Sampling a shared set of matrices from the base model...")
        base_model = load_model(args.model)
        selected_names = select_matrix_names(base_model, args.num_matrices, args.seed)
        del base_model

    with open(index_path, "a", encoding="utf-8") as index_handle:
        log.info("Running Wanda pruning...")
        wanda_model = load_model(args.model)
        recipe = WandaPruningModifier(
            sparsity=args.sparsity, mask_structure="0:0", targets="__ALL__"
        )
        oneshot(model=wanda_model, dataset=oneshot_dataset, recipe=recipe)
        rows_written = extract_weight_matrices(
            wanda_model,
            args.model,
            "wanda",
            args.sparsity,
            args.seed,
            selected_names,
            matrix_dir,
            index_handle,
            index_lock,
        )

        log.info("Running 2SSP pruning...")
        ssp_model = load_model(args.model)
        ssp_model.config.use_cache = False
        result = two_stage_2ssp(ssp_model, calibration_2ssp, args.sparsity)
        if result is False:
            log.error("2SSP pruning failed - invalid sparsity parameters")
        else:
            rows_written += extract_weight_matrices(
                ssp_model,
                args.model,
                "2ssp",
                args.sparsity,
                args.seed,
                selected_names,
                matrix_dir,
                index_handle,
                index_lock,
            )

    log.info(f"Wrote {rows_written} matrices to {matrix_dir}")
    log.info(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()

import argparse
import logging
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
from scipy.stats import ks_2samp, wasserstein_distance
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import get_dataset, get_text_from_item
from prune import get_tokenized_data
from similarity_check import prepare_calibration


FORMAT = "time=%(asctime)s level=%(levelname)s name=%(name)s msg=%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)
log = logging.getLogger(__name__)


def _get_dataset_split(raw_dataset):
    if isinstance(raw_dataset, dict):
        for split_name in ("train", "validation", "test"):
            if split_name in raw_dataset:
                return raw_dataset[split_name]
        return raw_dataset[list(raw_dataset.keys())[0]]
    return raw_dataset


def _materialize_samples(samples):
    if samples is None:
        return []
    return [samples[i] for i in range(len(samples))]


def _normalize_technique(name):
    aliases = {
        "random": "random_sample",
        "random_sample": "random_sample",
        "words_datasets": "words_dataset",
        "words_dataset": "words_dataset",
        "most_similar": "prototype",
        "prototype": "prototype",
    }
    return aliases.get(name, name)


def _texts_from_dataset(dataset, dataset_name, max_items):
    texts = []
    for index, item in enumerate(dataset):
        if index >= max_items:
            break
        texts.append(get_text_from_item(item, dataset_name))
    return texts


def _token_counts_from_texts(tokenizer, texts, max_length):
    counts = Counter()
    special_ids = set(tokenizer.all_special_ids or [])
    for text in texts:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        counts.update(token_id for token_id in token_ids if token_id not in special_ids)
    return counts


def _token_counts_from_samples(samples, tokenizer):
    counts = Counter()
    special_ids = set(tokenizer.all_special_ids or [])
    for sample in samples:
        input_ids = sample.get("input_ids")
        if input_ids is None:
            continue
        if isinstance(input_ids, torch.Tensor):
            token_ids = input_ids.detach().cpu().view(-1).tolist()
        else:
            token_ids = torch.as_tensor(input_ids).view(-1).tolist()
        counts.update(token_id for token_id in token_ids if token_id not in special_ids)
    return counts


def _distribution_metrics(target_counts, calib_counts, tail_fraction=0.1, smoothing=1e-12):
    support = sorted(set(target_counts) | set(calib_counts))
    if not support:
        return {
            "kl_target_to_calib": 0.0,
            "kl_calib_to_target": 0.0,
            "js_divergence": 0.0,
            "calib_entropy": 0.0,
            "unique_token_coverage": 0.0,
            "tail_token_coverage": 0.0,
            "kl_head": 0.0,
            "kl_tail": 0.0,
        }

    target = np.array([target_counts.get(token_id, 0) for token_id in support], dtype=np.float64)
    calib = np.array([calib_counts.get(token_id, 0) for token_id in support], dtype=np.float64)

    target = target + smoothing
    calib = calib + smoothing
    target = target / target.sum()
    calib = calib / calib.sum()

    kl_target_to_calib = float(scipy_entropy(target, calib))
    kl_calib_to_target = float(scipy_entropy(calib, target))
    mixture = 0.5 * (target + calib)
    js_divergence = 0.5 * float(scipy_entropy(target, mixture) + scipy_entropy(calib, mixture))
    calib_entropy = float(scipy_entropy(calib))

    target_types = set(target_counts)
    calib_types = set(calib_counts)
    unique_token_coverage = float(len(target_types & calib_types) / max(1, len(target_types)))

    if len(support) > 0 and tail_fraction > 0:
        target_freqs = np.array([target_counts[token_id] for token_id in support], dtype=np.float64)
        order = np.argsort(target_freqs)
        tail_size = max(1, int(np.ceil(len(support) * tail_fraction)))
        tail_tokens = {support[idx] for idx in order[:tail_size]}
        tail_token_coverage = float(len(tail_tokens & calib_types) / max(1, len(tail_tokens)))

        head_ids = {support[idx] for idx in order[-tail_size:]}
        tail_ids = {support[idx] for idx in order[:-tail_size]}

        def _kl_on_subset(ids):
            if not ids:
                return 0.0
            subset = sorted(ids)
            target_subset = np.array([target_counts.get(token_id, 0) for token_id in subset], dtype=np.float64)
            calib_subset = np.array([calib_counts.get(token_id, 0) for token_id in subset], dtype=np.float64)
            target_subset = target_subset + smoothing
            calib_subset = calib_subset + smoothing
            target_subset = target_subset / target_subset.sum()
            calib_subset = calib_subset / calib_subset.sum()
            return float(scipy_entropy(target_subset, calib_subset))

        kl_head = _kl_on_subset(head_ids)
        kl_tail = _kl_on_subset(tail_ids)
    else:
        tail_token_coverage = 0.0
        kl_head = 0.0
        kl_tail = 0.0

    return {
        "kl_target_to_calib": kl_target_to_calib,
        "kl_calib_to_target": kl_calib_to_target,
        "js_divergence": js_divergence,
        "calib_entropy": calib_entropy,
        "unique_token_coverage": unique_token_coverage,
        "tail_token_coverage": tail_token_coverage,
        "kl_head": kl_head,
        "kl_tail": kl_tail,
    }


def _batch_samples(samples, batch_size):
    batch = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _pad_and_stack(batch, key):
    tensors = []
    max_len = 0
    for sample in batch:
        value = sample.get(key)
        if value is None:
            continue
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.view(-1)
        tensors.append(tensor)
        max_len = max(max_len, tensor.numel())

    if not tensors:
        return None

    padded = []
    for tensor in tensors:
        if tensor.numel() < max_len:
            tensor = F.pad(tensor, (0, max_len - tensor.numel()), value=0)
        padded.append(tensor)
    return torch.stack(padded, dim=0)


def _activation_norms(model, samples, device, batch_size):
    model.eval()
    norms = []

    with torch.no_grad():
        for batch in _batch_samples(samples, batch_size):
            input_ids = _pad_and_stack(batch, "input_ids")
            if input_ids is None:
                continue

            attention_mask = _pad_and_stack(batch, "attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            norms.append(torch.norm(pooled.float(), p=2, dim=1).cpu())

    if not norms:
        return np.array([], dtype=np.float64)
    return torch.cat(norms, dim=0).numpy()


def _dataset_population_token_stats(tokenizer, dataset, dataset_name, max_population, max_length, seed):
    texts = _texts_from_dataset(dataset, dataset_name, max_population)
    rng = np.random.default_rng(seed)
    rng.shuffle(texts)
    return _token_counts_from_texts(tokenizer, texts, max_length=max_length), texts


def _save_dataset_plot(results_df, dataset_name, output_dir):
    if results_df.empty or "dataset" not in results_df.columns:
        return
    ds_df = results_df[results_df["dataset"] == dataset_name].copy()
    if ds_df.empty:
        return

    techniques = list(dict.fromkeys(ds_df["technique"].tolist()))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(techniques))))
    color_map = {technique: colors[index] for index, technique in enumerate(techniques)}

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for technique in techniques:
        tech_df = ds_df[ds_df["technique"] == technique].sort_values("nsamples")
        axes[0].plot(
            tech_df["nsamples"],
            tech_df["kl_target_to_calib"],
            marker="o",
            color=color_map[technique],
            label=technique,
        )
        axes[1].plot(
            tech_df["nsamples"],
            tech_df["activation_wasserstein"],
            marker="o",
            color=color_map[technique],
            label=technique,
        )

    axes[0].set_title(f"Token-distribution shift - {dataset_name}")
    axes[0].set_ylabel("KL(target || calib)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].set_title(f"Activation shift - {dataset_name}")
    axes[1].set_xlabel("Calibration samples")
    axes[1].set_ylabel("Wasserstein distance on pooled activation norms")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"theory_empirical_{dataset_name}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Empirical analysis for ZipCal theory support: token shift and activation shift"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["boolq", "winogrande", "arc_challenge"],
        help="Datasets to analyze",
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=["random_sample", "least_perplexity", "words_dataset"],
        help="Calibration techniques to compare",
    )
    parser.add_argument(
        "--nsamples_grid",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Calibration sample sizes for the ablation",
    )
    parser.add_argument("--max_population", type=int, default=2000, help="Maximum dataset items used to estimate the target distribution")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum token length used for token statistics")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for activation extraction")
    parser.add_argument("--output_csv", type=str, default="results/theory_empirical/theory_empirical_results.csv", help="Output CSV file")
    parser.add_argument("--output_dir", type=str, default="results/theory_empirical", help="Directory for plots")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.info(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    st_model = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    model_tag = args.model.replace("/", "-")
    rows = []

    for dataset_name in args.datasets:
        log.info(f"Analyzing dataset: {dataset_name}")
        raw_dataset = get_dataset(dataset_name)
        if raw_dataset is None:
            log.warning(f"Skipping {dataset_name}: dataset unavailable")
            continue

        dataset = _get_dataset_split(raw_dataset)
        if hasattr(dataset, "select"):
            population_dataset = dataset.select(range(min(len(dataset), args.max_population)))
        else:
            population_dataset = list(dataset)[: args.max_population]

        target_counts, texts = _dataset_population_token_stats(
            tokenizer,
            population_dataset,
            dataset_name,
            max_population=args.max_population,
            max_length=args.max_seq_len,
            seed=args.seed,
        )

        if not texts:
            log.warning(f"Skipping {dataset_name}: no texts available")
            continue

        population_tokenized = get_tokenized_data(
            population_dataset,
            tokenizer,
            dataset_name,
            max_length=args.max_seq_len,
            return_tensors=True,
        )

        population_activation_norms = _activation_norms(
            model,
            population_tokenized,
            device=device,
            batch_size=args.batch_size,
        )

        for nsamples in args.nsamples_grid:
            log.info(f"Dataset {dataset_name}: nsamples={nsamples}")
            for technique in args.techniques:
                normalized_technique = _normalize_technique(technique)
                log.info(f"Sampling technique: {technique} -> {normalized_technique}")
                try:
                    calibration = prepare_calibration(
                        model=st_model,
                        dataloader=[population_tokenized],
                        nsamples=nsamples,
                        type=normalized_technique,
                        tokenizer=tokenizer,
                        dataset_name=dataset_name,
                        model_name=model_tag,
                    )
                    calibration_samples = _materialize_samples(calibration)
                    calib_counts = _token_counts_from_samples(calibration_samples, tokenizer)
                    metrics = _distribution_metrics(target_counts, calib_counts)

                    calibration_activation_norms = _activation_norms(
                        model,
                        calibration_samples,
                        device=device,
                        batch_size=args.batch_size,
                    )

                    if len(population_activation_norms) and len(calibration_activation_norms):
                        activation_wasserstein = float(
                            wasserstein_distance(population_activation_norms, calibration_activation_norms)
                        )
                        activation_ks = float(
                            ks_2samp(population_activation_norms, calibration_activation_norms).statistic
                        )
                    else:
                        activation_wasserstein = 0.0
                        activation_ks = 0.0

                    rows.append(
                        {
                            "model": args.model,
                            "dataset": dataset_name,
                            "technique": technique,
                            "nsamples": nsamples,
                            "population_items": len(texts),
                            "calibration_items": len(calibration_samples),
                            "kl_target_to_calib": metrics["kl_target_to_calib"],
                            "kl_calib_to_target": metrics["kl_calib_to_target"],
                            "js_divergence": metrics["js_divergence"],
                            "calib_entropy": metrics["calib_entropy"],
                            "unique_token_coverage": metrics["unique_token_coverage"],
                            "tail_token_coverage": metrics["tail_token_coverage"],
                            "kl_head": metrics["kl_head"],
                            "kl_tail": metrics["kl_tail"],
                            "activation_wasserstein": activation_wasserstein,
                            "activation_ks": activation_ks,
                            "population_activation_norm_mean": float(np.mean(population_activation_norms)) if len(population_activation_norms) else 0.0,
                            "calibration_activation_norm_mean": float(np.mean(calibration_activation_norms)) if len(calibration_activation_norms) else 0.0,
                        }
                    )
                except Exception as exc:
                    log.warning(f"Failed for dataset={dataset_name}, technique={technique}, nsamples={nsamples}: {exc}")

        dataset_rows = pd.DataFrame(rows)
        _save_dataset_plot(dataset_rows, dataset_name, args.output_dir)

    if not rows:
        log.warning("No results were produced.")
        return

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    log.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
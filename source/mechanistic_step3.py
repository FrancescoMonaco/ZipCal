import argparse
import os
import torch
import pandas as pd
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import get_dataset
from similarity_check import prepare_calibration
from prune import get_tokenized_data
from first_layers_analysis import get_layers_wanda_metrics, compute_mask_overlap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def get_calib_data(d_name, tokenizer, nsamples, calib_type):
    raw_dataset = get_dataset(d_name)
    dataset = raw_dataset["train"] if isinstance(raw_dataset, dict) and "train" in raw_dataset else raw_dataset
    if isinstance(dataset, dict) and "validation" in dataset: dataset = dataset["validation"]
    tokenized = get_tokenized_data(dataset, tokenizer, d_name, return_tensors=True)
    class DummyModel:
        def __init__(self): self.device = "cpu"
    calib = prepare_calibration(model=DummyModel(), dataloader=[tokenized], nsamples=nsamples, type=calib_type, tokenizer=tokenizer, dataset_name=d_name, model_name="dummy")
    return [calib[i] for i in range(len(calib))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--oracle_samples", type=int, default=1024)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--n_layers", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    logging.info("Getting Oracle Calib...")
    oracle_calib = get_calib_data(args.dataset, tokenizer, args.oracle_samples, "random_sample")

    logging.info("Getting Random Calib...")
    random_calib = get_calib_data(args.dataset, tokenizer, args.nsamples, "random_sample")

    logging.info("Getting Words Calib...")
    words_calib = get_calib_data(args.dataset, tokenizer, args.nsamples, "words_dataset")

    logging.info("Metrics for Oracle...")
    metrics_oracle = get_layers_wanda_metrics(model, oracle_calib, start_layer=0, end_layer=args.n_layers, device="cuda")
    
    logging.info("Metrics for Random...")
    metrics_rand = get_layers_wanda_metrics(model, random_calib, start_layer=0, end_layer=args.n_layers, device="cuda")

    logging.info("Metrics for Words...")
    metrics_words = get_layers_wanda_metrics(model, words_calib, start_layer=0, end_layer=args.n_layers, device="cuda")

    logging.info("Computing overlaps vs Oracle...")
    overlap_rand = compute_mask_overlap(metrics_oracle, metrics_rand, args.sparsity)
    overlap_words = compute_mask_overlap(metrics_oracle, metrics_words, args.sparsity)

    results = []
    for layer in overlap_rand:
        results.append({
            "model": args.model, "dataset": args.dataset, "layer": layer,
            "overlap_random": overlap_rand[layer], "overlap_words": overlap_words[layer]
        })
        logging.info(f"Layer {layer}: Random Overlap = {overlap_rand[layer]:.4f}, Words Overlap = {overlap_words[layer]:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("results/mechanistic_step3_overlap.csv", index=False)

if __name__ == "__main__":
    main()

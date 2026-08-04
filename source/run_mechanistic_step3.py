import argparse
import os
import torch
import pandas as pd
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from data import get_dataset
from similarity_check import prepare_calibration
from prune import get_tokenized_data
from first_layers_analysis import get_layers_wanda_metrics, compute_mask_overlap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def get_calib_data(d_name, tokenizer, nsamples, calib_type, total_samples=False, st_model=None):
    # Usa le tue utility predefinite che caricano correttamente split, configurazioni e mappe
    dataset = get_dataset(d_name, split="train")
    if dataset is None:
        dataset = get_dataset(d_name, split="validation")
        
    tokenized = get_tokenized_data(dataset, tokenizer, d_name, return_tensors=True)
    
    if total_samples:
        nsamples = min(len(dataset), 1024) # Oracle max limit 1024 per evitare OOM
        calib_type = "random_sample"
        
    class DummyModel:
        def __init__(self): self.device = "cpu"
        
    calib = prepare_calibration(
        model=st_model if st_model else DummyModel(), 
        dataloader=[tokenized], 
        nsamples=nsamples, 
        type=calib_type, 
        tokenizer=tokenizer, 
        dataset_name=d_name, 
        model_name="llama-3.1-8b"
    )
    return [calib[i] for i in range(len(calib))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    # Dataset specificati richiesti
    parser.add_argument("--datasets", type=str, nargs="+", default=["hellaswag", "boolq", "gsm8k"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--n_layers", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    # st_model serve a prepare_calibration come dependency potenziale
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    results = []

    for d_name in args.datasets:
        logging.info(f"--- Processing Dataset: {d_name} ---")
        
        logging.info(f"Getting Oracle Calib per {d_name}..")
        oracle_calib = get_calib_data(d_name, tokenizer, args.nsamples, "random_sample", total_samples=True, st_model=st_model)

        logging.info(f"Getting Monolithic Random Sample Calib ({args.nsamples} samples)...")
        random_calib = get_calib_data(d_name, tokenizer, args.nsamples, "random_sample", total_samples=False, st_model=st_model)

        logging.info(f"Getting Lexical Diversity Words Dataset Calib ({args.nsamples} samples)...")
        words_calib = get_calib_data(d_name, tokenizer, args.nsamples, "words_dataset", total_samples=False, st_model=st_model)

        logging.info("Metrics calcolo su Oracle...")
        metrics_oracle = get_layers_wanda_metrics(model, oracle_calib, start_layer=0, end_layer=args.n_layers, device="cuda")
        
        logging.info("Metrics calcolo su Random...")
        metrics_rand = get_layers_wanda_metrics(model, random_calib, start_layer=0, end_layer=args.n_layers, device="cuda")

        logging.info("Metrics calcolo su Lexical Diversity (words_dataset)...")
        metrics_words = get_layers_wanda_metrics(model, words_calib, start_layer=0, end_layer=args.n_layers, device="cuda")

        logging.info("Computing overlap vs Oracle...")
        overlap_rand = compute_mask_overlap(metrics_oracle, metrics_rand, args.sparsity)
        overlap_words = compute_mask_overlap(metrics_oracle, metrics_words, args.sparsity)

        for layer in overlap_words:
            results.append({
                "model": args.model, 
                "dataset": d_name, 
                "layer": layer,
                "overlap_random": overlap_rand[layer],
                "overlap_words": overlap_words[layer],
                "n_samples_calibration": args.nsamples,
                "oracle_samples": len(oracle_calib)
            })
            logging.info(f"[{d_name} | Layer {layer}] RND Overlap: {overlap_rand[layer]:.4f} | VOCAB Overlap: {overlap_words[layer]:.4f}")

    df = pd.DataFrame(results)
    os.makedirs("results/theory_empirical", exist_ok=True)
    out_file = "results/theory_empirical/mechanistic_overlap.csv"
    df.to_csv(out_file, index=False)
    logging.info(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()

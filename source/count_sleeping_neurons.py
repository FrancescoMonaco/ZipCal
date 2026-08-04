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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def get_calib_data(d_name, tokenizer, nsamples, calib_type, st_model):
    dataset = get_dataset(d_name, split="train")
    if dataset is None:
        dataset = get_dataset(d_name, split="validation")
        
    tokenized = get_tokenized_data(dataset, tokenizer, d_name, return_tensors=True)
    
    class DummyModel:
        def __init__(self): self.device = "cpu"
        
    calib = prepare_calibration(
        model=st_model, 
        dataloader=[tokenized], 
        nsamples=nsamples, 
        type=calib_type, 
        tokenizer=tokenizer, 
        dataset_name=d_name, 
        model_name="llama-3.1-8b"
    )
    return [calib[i] for i in range(len(calib))]

def count_sleeping_neurons(model, calib_samples, n_layers=5, device="cuda"):
    layers = model.model.layers
    act_counts = {}
    handles = []

    def get_hook(name):
        def hook(module, inp, out):
            # inp[0] shape: (batch, seq_len, intermediate_size)
            # We want to know if the inputs to down_proj (the SwiGLU activations) are 0
            x = inp[0].detach().view(-1, inp[0].shape[-1])
            # Count how many times each neuron has a non-negligible activation
            active_count = (x.abs() > 1e-6).sum(dim=0).float()
            if name not in act_counts:
                act_counts[name] = active_count
            else:
                act_counts[name] += active_count
        return hook

    # Target down_proj in the first few layers where LLMs map vocabulary patterns
    for idx in range(n_layers):
        name = f"layer_{idx}_mlp.down_proj"
        handles.append(layers[idx].mlp.down_proj.register_forward_hook(get_hook(name)))

    model.eval()
    with torch.no_grad():
        for item in calib_samples:
            input_ids = item["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
            model(input_ids.to(device))

    for h in handles:
        h.remove()

    summary = {}
    for name, counts in act_counts.items():
        # A neuron is considered "sleeping" if it literally never fired across the whole calib set
        sleeping = (counts == 0).sum().item()
        total = counts.numel()
        summary[name] = {
            "sleeping_neurons": sleeping, 
            "total_neurons": total, 
            "sleeping_ratio": sleeping / total
        }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--datasets", type=str, nargs="+", default=["hellaswag", "boolq", "gsm8k"])
    parser.add_argument("--nsamples", type=int, default=128)
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
    
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    results = []

    for d_name in args.datasets:
        logging.info(f"--- Dataset: {d_name} ---")

        logging.info(f"Testing Random Sample...")
        random_calib = get_calib_data(d_name, tokenizer, args.nsamples, "random_sample", st_model)
        rand_sleep = count_sleeping_neurons(model, random_calib, n_layers=args.n_layers)

        logging.info(f"Testing Words Dataset (Lexical Diversity)...")
        words_calib = get_calib_data(d_name, tokenizer, args.nsamples, "words_dataset", st_model)
        words_sleep = count_sleeping_neurons(model, words_calib, n_layers=args.n_layers)

        for layer_name in rand_sleep:
            r_sleep = rand_sleep[layer_name]["sleeping_neurons"]
            w_sleep = words_sleep[layer_name]["sleeping_neurons"]
            tot = rand_sleep[layer_name]["total_neurons"]
            
            logging.info(f"[{d_name} | {layer_name}] Sleeping: {r_sleep} (RND) vs {w_sleep} (Words)")
            
            results.append({
                "dataset": d_name,
                "layer": layer_name,
                "random_sleeping": r_sleep,
                "words_sleeping": w_sleep,
                "random_ratio": rand_sleep[layer_name]["sleeping_ratio"],
                "words_ratio": words_sleep[layer_name]["sleeping_ratio"],
                "total_neurons": tot
            })

    df = pd.DataFrame(results)
    os.makedirs("results/theory_empirical", exist_ok=True)
    out_file = "results/theory_empirical/sleeping_neurons.csv"
    df.to_csv(out_file, index=False)
    logging.info(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()

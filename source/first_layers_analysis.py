import argparse
import os
import torch
import pandas as pd
import logging
import copy
from filelock import FileLock
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import torch.nn as nn
from data import get_dataset, get_text_from_item
from similarity_check import prepare_calibration
from prune import get_tokenized_data
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class WrappedGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            tmp = inp.shape[0] * inp.shape[1]
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

def get_layers_wanda_metrics(model, calib_samples, start_layer=0, end_layer=3, device="cuda"):
    """
    Get Wanda metrics for a specific range of layers.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        return {}

    total_layers = len(layers)
    # Ensure indices are valid
    start_layer = max(0, start_layer)
    end_layer = min(total_layers, end_layer)
    target_layer_indices = list(range(start_layer, end_layer))
    
    log.info(f"Analyzing layers: {target_layer_indices}")

    wrapped_modules = {}
    handles = []
    
    def get_hook(name):
        def hook(module, inp, out):
            if name not in wrapped_modules:
                wrapped_modules[name] = WrappedGPT(module)
            wrapped_modules[name].add_batch(inp[0].data, out.data)
        return hook

    for idx in target_layer_indices:
        layer = layers[idx]
        subset = find_layers(layer)
        for name, module in subset.items():
            full_name = f"layer_{idx}_{name}"
            handles.append(module.register_forward_hook(get_hook(full_name)))

    model.eval()
    with torch.no_grad():
        for item in calib_samples:
            input_ids = item["input_ids"]
            attention_mask = item.get("attention_mask")
            if not isinstance(input_ids, torch.Tensor):
                input_ids_t = torch.tensor(input_ids).to(device)
            else:
                input_ids_t = input_ids.to(device)
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask_t = torch.tensor(attention_mask).to(device)
                else:
                    attention_mask_t = attention_mask.to(device)
            else:
                attention_mask_t = None
            
            if len(input_ids_t.shape) == 1:
                input_ids_t = input_ids_t.unsqueeze(0)
            if attention_mask_t is not None and len(attention_mask_t.shape) == 1:
                attention_mask_t = attention_mask_t.unsqueeze(0)
                
            model(input_ids_t, attention_mask=attention_mask_t)

    for h in handles:
        h.remove()

    metrics = {}
    for idx in target_layer_indices:
        for full_name, wrapped in wrapped_modules.items():
            if full_name.startswith(f"layer_{idx}_"):
                # Wanda Score: |W| * sqrt(S)
                W_metric = torch.abs(wrapped.layer.weight.data) * torch.sqrt(wrapped.scaler_row.reshape((1, -1)))
                metrics[full_name] = W_metric.cpu()
    return metrics

def compute_mask_overlap(metrics1, metrics2, sparsity):
    overlaps = {}
    for name in metrics1:
        if name not in metrics2: continue
        
        m1 = metrics1[name]
        m2 = metrics2[name]
        
        rows, cols = m1.shape
        k = int(cols * (1 - sparsity))
        if k == 0: k = 1
        
        _, top_idx1 = torch.topk(m1, k, dim=1)
        _, top_idx2 = torch.topk(m2, k, dim=1)
        
        row_offsets = torch.arange(rows, device=m1.device).unsqueeze(1) * cols
        flat_idx1 = (top_idx1 + row_offsets).view(-1)
        flat_idx2 = (top_idx2 + row_offsets).view(-1)
        
        # Intersection
        # We can use np.intersect1d which is slow on gpu tensors, or remain on cpu
        # Indices are on CPU already from get_layers_wanda_metrics
        
        # Set based intersection for per-row indices
        # Since we just need count, and we know they are unique per row...
        # Let's just do Python sets for simplicity or broadcasting
        
        # Faster way on CPU tensors:
        t1 = torch.zeros(rows * cols, dtype=torch.bool)
        t1[flat_idx1] = True
        t2 = torch.zeros(rows * cols, dtype=torch.bool)
        t2[flat_idx2] = True
        
        intersection = (t1 & t2).sum().item()
        total_pruned = rows * k 
        
        overlap_ratio = intersection / total_pruned
        overlaps[name] = overlap_ratio
        
    return overlaps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-7b")
    parser.add_argument("--datasets", type=str, nargs="+", default=["c4"], help="Datasets for calibration")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--n_first_layers", type=int, default=5, help="Number of first layers to analyze")
    parser.add_argument("--output_csv", type=str, default="results/first_layers_overlap.csv")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load Model
    log.info(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype="auto", 
        device_map=args.device,
        trust_remote_code=True
    )
    
    # Sentence Transformer for calibration preparation (if needed by similarity, but reusing prepare_calibration)
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu") # Keep on CPU to save VRAM

    calibration_datasets = args.datasets
    
    results = []

    for d_name in calibration_datasets:
        log.info(f"Processing {d_name}...")
        
        raw_dataset = get_dataset(d_name)
        if raw_dataset is None:
            continue
        
        if isinstance(raw_dataset, dict):
             if "train" in raw_dataset:
                dataset = raw_dataset["train"]
             elif "validation" in raw_dataset:
                dataset = raw_dataset["validation"]
             else:
                 dataset = list(raw_dataset.values())[0]
        else:
            dataset = raw_dataset
            
        tokenized = get_tokenized_data(dataset, tokenizer, d_name, return_tensors=True)
        current_tokenized_data = [Dataset.from_list(tokenized)]
        
        # Get Random Samples
        log.info(f"Sampling Random for {d_name}")
        rand_calib = prepare_calibration(
            model=st_model,
            dataloader=current_tokenized_data,
            nsamples=args.nsamples,
            type="random_sample",
            tokenizer=tokenizer,
            dataset_name=d_name,
            model_name=args.model.replace("/", "-")
        )
        rand_list = [rand_calib[i] for i in range(len(rand_calib))]
        
        # Get Unique Tokens (Random Words)
        log.info(f"Sampling Unique Tokens for {d_name}")
        unique_calib = prepare_calibration(
            model=st_model,
            dataloader=current_tokenized_data,
            nsamples=args.nsamples,
            type="words_dataset", # This triggers the unique token logic in prepare_calibration
            tokenizer=tokenizer,
            dataset_name=d_name,
            model_name=args.model.replace("/", "-")
        )
        unique_list = [unique_calib[i] for i in range(len(unique_calib))]
        
        # Calculate Metrics for First Layers
        log.info(f"Helper metrics for first {args.n_first_layers} layers...")
        metrics_rand = get_layers_wanda_metrics(model, rand_list, start_layer=0, end_layer=args.n_first_layers, device=args.device)
        metrics_unique = get_layers_wanda_metrics(model, unique_list, start_layer=0, end_layer=args.n_first_layers, device=args.device)
        
        # Compute Overlap
        mask_overlaps = compute_mask_overlap(metrics_rand, metrics_unique, args.sparsity)
        
        # Per layer overlap
        for layer_name, overlap in mask_overlaps.items():
            run_data = {
                "model": args.model,
                "dataset": d_name,
                "layer": layer_name,
                "overlap": overlap,
                "n_samples": args.nsamples,
                "sparsity": args.sparsity
            }
            results.append(run_data)
            
        mean_overlap = sum(mask_overlaps.values()) / len(mask_overlaps) if mask_overlaps else 0
        log.info(f"[{d_name}] Mean mask overlap (first {args.n_first_layers} layers): {mean_overlap:.4f}")
        
    df = pd.DataFrame(results)
    if not df.empty:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with FileLock(args.output_csv + ".lock"):
            df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)
        log.info(f"Results saved to {args.output_csv}")
    else:
        log.info("No results to save.")

if __name__ == "__main__":
    main()

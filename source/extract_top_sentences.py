import sys
import torch
import statistics
import collections
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import get_dataset, get_text_from_item
from similarity_check import zipf_sampling, least_perplexity_sampling

def get_text_length_and_lexical_stats(texts):
    lengths = [len(t.split()) for t in texts if t.strip()]
    if not lengths:
        return 0, 0, 0, 0, set()
    
    unique_words = set()
    total_words = 0
    word_freq = collections.Counter()
    for t in texts:
        words = t.split()
        total_words += len(words)
        unique_words.update(words)
        word_freq.update(words)
        
    min_l = min(lengths)
    max_l = max(lengths)
    med_l = statistics.median(lengths)
    
    return min_l, max_l, med_l, len(unique_words) / max(1, total_words), unique_words

def get_tokenized_data(dataset, tokenizer, dataset_name, max_length=1024, return_tensors=True):
    processed_dataset = []
    for item in dataset:
        text = get_text_from_item(item, dataset_name)
        encoded = tokenizer(text, truncation=True, max_length=max_length)
        if return_tensors:
            encoded = tokenizer.pad(
                encoded, padding="max_length", return_tensors="pt", max_length=max_length
            )
            processed_dataset.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "text": text,
                }
            )
    return processed_dataset

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    datasets_to_test = ["winogrande", "hellaswag", "boolq"]
    total_samples = 16
    print_samples = 3
    
    for ds_name in datasets_to_test:
        print(f"\n==============================")
        print(f"Dataset: {ds_name}")
        print(f"==============================")
        try:
            ds = get_dataset(ds_name.replace("/", "___"), split="train")
            # Limit elements for speed if necessary, but sampling usually takes a dataloader
            # We will use the first 1000 items to speed up
            ds = [ds[i] for i in range(min(1000, len(ds)))]
        except Exception as e:
            print(f"Could not load {ds_name}: {e}")
            continue
            
        tokenized_data = get_tokenized_data(ds, tokenizer, ds_name.replace("/", "___"), max_length=2048, return_tensors=True)
        # Put in dataloader format (list of lists)
        dataloader = [tokenized_data]
        
        # Calculate dataset stats
        all_texts = [item["text"] for item in tokenized_data]
        d_min, d_max, d_med, d_lex, _ = get_text_length_and_lexical_stats(all_texts)
        print(f"Dataset Stats ({len(all_texts)} items) - Length (words): Min={d_min}, Max={d_max}, Median={d_med}")
        print(f"Dataset Type-Token Ratio: {d_lex:.4f}")
        
        z_vocab = set()
        print("\n--- Zipf Sampling (Ours) ---")
        try:
            zipf_samples = zipf_sampling(dataloader, total_samples, tokenizer=tokenizer, shuffle=False)
            zipf_texts = []
            for i, item in enumerate(zipf_samples):
                text = item.get("text", "")
                if not text:
                    text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
                zipf_texts.append(text)
                if i < print_samples:
                    print(f"{i+1}. {text}")
                
            z_min, z_max, z_med, z_lex, z_vocab = get_text_length_and_lexical_stats(zipf_texts)
            print(f"> Zipf Stats ({total_samples} items) - Length: Min={z_min}, Max={z_max}, Median={z_med}")
            print(f"> Zipf Type-Token Ratio: {z_lex:.4f}, Unique Words: {len(z_vocab)}")
        except Exception as e:
            print(f"Zipf Sampling failed: {e}")
            
        c_vocab = set()
        print("\n--- Least Perplexity Sampling (COLA) ---")
        try:
            cola_samples = least_perplexity_sampling(
                dataloader, model, total_samples, tokenizer, return_distribution=False
            )
            cola_texts = []
            for i, item in enumerate(cola_samples):
                text = item.get("text", "")
                if not text:
                    text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
                cola_texts.append(text)
                if i < print_samples:
                    print(f"{i+1}. {text}")
                
            c_min, c_max, c_med, c_lex, c_vocab = get_text_length_and_lexical_stats(cola_texts)
            print(f"> COLA Stats ({total_samples} items) - Length: Min={c_min}, Max={c_max}, Median={c_med}")
            print(f"> COLA Type-Token Ratio: {c_lex:.4f}, Unique Words: {len(c_vocab)}")
        except Exception as e:
            print(f"Least Perplexity Sampling failed: {e}")

        # Intersection difference
        if z_vocab and c_vocab:
            print(f"\n> Lexical Differences between Samples:")
            print(f"Words only in Zipf: {len(z_vocab - c_vocab)}")
            print(f"Words only in COLA: {len(c_vocab - z_vocab)}")
            print(f"Intersection: {len(z_vocab.intersection(c_vocab))}")

if __name__ == "__main__":
    main()

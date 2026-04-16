"""
Datasets
"""

import os
from pathlib import Path
from datasets import load_dataset


# Datasets to be used in the project
general_datasets = ["c4", "oscar", "wikitext", "pile"]
aritm_reasoning = ["gsm8k", "svamp"]
nlu_inference = ["anli_r1", "rte"]
commonsense_qa = ["boolq", "commonsense_qa", "race", "winogrande", "hellaswag", "mmlu"]
translation = ["wmt14",]
# coding = ["opc", "ds1000", "mbpp"]  # rimossi perché senza split train/test
mixed = ["c4", "gsm8k", "anli_r1", "boolq", "wmt14"]
all_datasets = (
    general_datasets
    + aritm_reasoning
    + nlu_inference
    + commonsense_qa
    + translation
    + mixed
)

# Datasets should be in the datasets/ folder
DATASETS_PATH = Path(os.path.dirname(__file__)).parent / "datasets"
os.makedirs(DATASETS_PATH, exist_ok=True)

# Mapping from local name to HuggingFace dataset ID (and optional config)
DATASET_MAPPING = {
    # Solo una piccola parte di C4 per evitare download enormi
    "c4": ("allenai/c4", "en"),
    "wikitext": ("wikitext", "wikitext-103-raw-v1"),
    "oscar": ("oscar-corpus/OSCAR-2301", "en"),
    "oscar_es": ("oscar-corpus/OSCAR-2301", "es"),
    "oscar_zh": ("oscar-corpus/OSCAR-2301", "zh"),
    "mc4_es": ("mc4", "es"),
    "mc4_zh": ("mc4", "zh"),
    # Solo una piccola parte di pile per evitare download enormi
    "pile": ("monology/pile-uncopyrighted", None),
    "gsm8k": ("gsm8k", "main"),
    "svamp": ("ChilleD/SVAMP", None),
    "anli_r1": ("facebook/anli", None),
    "rte": ("aps/super_glue", "rte"),
    "boolq": ("google/boolq", None),
    "commonsense_qa": ("tau/commonsense_qa", None),
    "race": ("race", "all"),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "wmt14": ("wmt/wmt14", "de-en"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge"),
    "hellaswag": ("Rowan/hellaswag", None),
    "mmlu": ("cais/mmlu", "all"),
    "xnli_es": ("xnli", "es"),
    "xnli_zh": ("xnli", "zh"),
    "xwinograd_zh": ("Muennighoff/xwinograd", "zh"),
    "xcopa_zh": ("xcopa", "zh"),
    "xquad_es": ("xquad", "xquad.es"),
    "xquad_zh": ("xquad", "xquad.zh"),
}


def get_text_from_item(item, dataset_name):
    """
    Extract text from a dataset item based on the dataset name.
    """
    base_name = dataset_name.split("_", 1)[0]

    if dataset_name in ["c4", "oscar", "redpajama", "pile", "mbpp"]:
        return item.get("text", "")
    elif base_name in ["oscar", "mc4"]:
        return item.get("text", "")
    elif dataset_name == "gsm8k":
        return item.get("question", "")
    elif dataset_name == "svamp":
        return item.get("Body", "") + " " + item.get("Question", "")
    elif dataset_name == "mawps":
        return item.get("sQuestion", "")
    elif dataset_name in ["anli_r1", "esnli", "rte"] or base_name == "xnli":
        return item.get("premise", "") + " " + item.get("hypothesis", "")
    elif dataset_name == "boolq":
        return item.get("passage", "") + " " + item.get("question", "")
    elif dataset_name == "commonsense_qa":
        return item.get("question", "")
    elif dataset_name == "race":
        return item.get("article", "") + " " + item.get("question", "")
    elif dataset_name == "winogrande":
        return item.get("sentence", "")
    elif base_name == "xwinograd":
        return item.get("sentence", "")
    elif base_name == "xcopa":
        return (
            item.get("premise", "")
            + " "
            + item.get("choice1", "")
            + " "
            + item.get("choice2", "")
        )
    elif base_name == "xquad":
        return item.get("context", "") + " " + item.get("question", "")
    elif dataset_name == "hellaswag":
        return item.get("ctx", "") + " " + item.get("activity_label", "")
    elif dataset_name == "arc_challenge":
        return item.get("question", "") + " " + " ".join(item.get("choices", {}).get("text", []))
    elif dataset_name == "wmt14":
        return item.get("translation", {}).get("en", "")
    elif dataset_name == "iwslt":
        return item.get("translation", {}).get("en", "")
    elif dataset_name in ["opc", "ds1000"]:
        return item.get("prompt", "")
    else:
        # Fallback strategies
        if "text" in item:
            return item["text"]
        if "sentence" in item:
            return item["sentence"]
        if "prompt" in item:
            return item["prompt"]
        if "question" in item:
            return item["question"]
        return str(item)


def get_dataset(name, split=None):
    """
    Load a dataset by name. Downloads it to DATASETS_PATH if not present.

    Args:
        name (str): Name of the dataset (e.g., 'c4', 'gsm8k').
        split (str, optional): Split to load (e.g., 'train', 'validation').

    Returns:
        Dataset or DatasetDict: The loaded dataset.
    """
    if name not in DATASET_MAPPING:
        hf_name = name
        config = None
    else:
        hf_name, config = DATASET_MAPPING[name]

    print(f"Loading dataset: {name} (HF: {hf_name}, Config: {config})")

    def _find_local_files(subpath):
        p = DATASETS_PATH / subpath
        if not p.exists():
            return []
        patterns = ["**/*.json*", "**/*.jsonl*", "**/*.parquet", "**/*.parquet.zst", "**/*.zst"]
        files = []
        for pat in patterns:
            files.extend(list(p.glob(pat)))
        # deduplicate and sort
        files = sorted(list({str(f): f for f in files}.values()))
        return files

    def _choose_format(files):
        # prefer parquet if any parquet present
        for f in files:
            if ".parquet" in str(f):
                return "parquet"
        return "json"

    try:
        # Prefer local files for C4
        if name == "c4":
            files = _find_local_files("c4")
            if not files:
                print(f"Local file for C4 split '{split}' not found.")
                return None
            # try to pick files matching the requested split name
            split_files = [f for f in files if split and split in str(f)] if split else files
            if not split_files and split:
                # fallback: pick any file and let HF create the split
                split_files = files[:2]
            fmt = _choose_format(split_files)
            dataset = load_dataset(fmt, data_files={split: [str(f) for f in split_files]} if split else {"train": [str(f) for f in split_files]}, split=split)
            return dataset

        # Prefer local files for Pile
        if name == "pile":
            files = _find_local_files("pile")
            if not files:
                print(f"Local file for Pile split '{split}' not found.")
                return None
            split_files = [f for f in files if split and split in str(f)] if split else files
            if not split_files:
                split_files = files[:2]
            fmt = _choose_format(split_files)
            dataset = load_dataset(fmt, data_files={split: [str(f) for f in split_files]} if split else {"train": [str(f) for f in split_files]}, split=split)
            return dataset

        # Try loading from the Hub; if split is unknown, attempt fallback to a matching variant
        try:
            if config:
                dataset = load_dataset(hf_name, config, cache_dir=str(DATASETS_PATH), split=split)
            else:
                dataset = load_dataset(hf_name, cache_dir=str(DATASETS_PATH), split=split)
            return dataset
        except Exception as e:
            err = str(e)
            # handle unknown split variants (e.g., anli has train_r1, train_r2...)
            if "Unknown split" in err or "Should be one of" in err:
                # load without split to inspect available splits
                if config:
                    ds_dict = load_dataset(hf_name, config, cache_dir=str(DATASETS_PATH))
                else:
                    ds_dict = load_dataset(hf_name, cache_dir=str(DATASETS_PATH))
                # map requested split to an available variant
                desired = split or "train"
                keys = list(ds_dict.keys())
                target = None
                if desired in ["train", "validation", "test"]:
                    for k in keys:
                        if k.startswith(desired) or desired in k:
                            target = k
                            break
                if target is None:
                    # fallback to first available split
                    target = keys[0]
                print(f"Using fallback split '{target}' for dataset {name} (requested '{split}').")
                return ds_dict[target]
            else:
                print(f"Error loading dataset {name}: {e}")
                return None
    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        return None


if __name__ == "__main__":
    # Test loading for all datasets with train/test split
    test_split = "test"
    train_split = "train"
    for ds_name in DATASET_MAPPING.keys():
        print(f"\nTesting dataset: {ds_name}")
        try:
            ds_train = get_dataset(ds_name, split=train_split)
            ds_test = get_dataset(ds_name, split=test_split)
            if ds_train is not None:
                print(f"  Train split loaded: {len(ds_train)} examples.")
            else:
                print("  Train split NOT available.")
            if ds_test is not None:
                print(f"  Test split loaded: {len(ds_test)} examples.")
            else:
                print("  Test split NOT available.")
        except Exception as e:
            print(f"  Error loading {ds_name}: {e}")

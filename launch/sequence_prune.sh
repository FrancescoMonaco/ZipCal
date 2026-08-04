#!/usr/bin/env bash
set -euo pipefail

# Initialize micromamba (robust): prefer PATH, fallback to common locations

MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it")
TASKS=("boolq" "winogrande" "arc_challenge" "gsm8k" "hellaswag")
TYPES=("random" "words_dataset")
NSAMPLES=(32 64 128)

mkdir -p logs

for model in "${MODELS[@]}"; do
  mname=$(echo "$model" | sed 's#/#!-#g' | sed 's#/#-#g')
  for task in "${TASKS[@]}"; do
    for t in "${TYPES[@]}"; do
      for n in "${NSAMPLES[@]}"; do
        logfile="logs/prune_${mname}_${task}_${t}_${n}.log"
        echo "Running: model=${model} task=${task} type=${t} nsamples=${n} -> ${logfile}"
        python source/prune.py --pruning_type ${t} --datasets ${task} --model ${model} --nsamples ${n} --run_pruning &> "${logfile}"
        echo "Finished: ${logfile}"
      done
    done
  done
done

echo "All jobs finished." 

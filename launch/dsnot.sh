#!/bin/bash
set -e

TOTAL_GPUS=4        # GPU fisiche dedicate a cola.sh (es. 2-3)
GPUS_PER_JOB=2      # assegna due GPU a ogni processo eval_cola
MAX_JOBS=$((TOTAL_GPUS / GPUS_PER_JOB))
GPU_OFFSET=0        # offset: usa le GPU 2-3 quando 0-1 sono occupate da eval.sh
declare -a PIDS=()  # Tracciamento dei PID in background

# Restituisce il primo slot libero, oppure -1 se tutti occupati
find_free_slot() {
    for ((i=0; i<MAX_JOBS; i++)); do
        if [[ -z "${PIDS[$i]}" ]] || ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "$i"
            return
        fi
    done
    echo "-1"
}

# Attende finché uno slot GPU non è libero e restituisce l'indice dello slot
acquire_gpu() {
    while true; do
        local slot
        slot=$(find_free_slot)
        if [[ "$slot" != "-1" ]]; then
            echo "$slot"
            return
        fi
        # Tutti gli slot sono occupati: attendi e riprova
        sleep 5
    done
}

# Lancia un job su uno specifico slot GPU in background
launch_job() {
    local SLOT_ID=$1; shift
    local START_GPU=$((GPU_OFFSET + SLOT_ID * GPUS_PER_JOB))
    local END_GPU=$((START_GPU + GPUS_PER_JOB - 1))
    local DEVICES
    DEVICES=$(seq -s, "$START_GPU" "$END_GPU")

    CUDA_VISIBLE_DEVICES=$DEVICES python source/eval_dsnot.py "$@" &
    PIDS[$SLOT_ID]=$!
}

DATASET_PREF="--datasets"
DATASETS=("winogrande" "arc_challenge" "boolq" "hellaswag" "openbookqa" "rte" "mmlu" "wmt14" "anli_r1" "svamp" "gsm8k" "pile" "wikitext" "c4" "winogrande arc_challenge boolq hellaswag openbookqa rte")
MODEL_PREF="--model"
MODELS=("meta-llama/Llama-3.1-70B-Instruct") #"meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it")
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128)
COMPRESSION_PREF="--compression_type"
COMPRESSION_TYPES=("pruning" "quantization" "awq" "2ssp")
OUTPUT_CSV_PREF="--output_csv"
OUTPUT_CSV="results/dsnot_experiments.csv"

mkdir -p logs

TASK_ID=0
# Gerarchia: Model -> Dataset -> Sample -> Compression
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for NSAMPLES in "${NUM_SAMPLES[@]}"; do
            for COMPRESSION in "${COMPRESSION_TYPES[@]}"; do
                GPU_SLOT=$(find_free_slot)
                if [[ "$GPU_SLOT" == "-1" ]]; then
                    GPU_SLOT=$(acquire_gpu)
                fi

                START_GPU=$((GPU_OFFSET + GPU_SLOT * GPUS_PER_JOB))
                END_GPU=$((START_GPU + GPUS_PER_JOB - 1))
                GPU_SET=$(seq -s, "$START_GPU" "$END_GPU")

                echo "================================================================"
                echo "TASK $TASK_ID -> GPU slot $GPU_SLOT (CUDA_VISIBLE_DEVICES=$GPU_SET)"
                echo "Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES, Compression: $COMPRESSION"
                echo "================================================================"

                LOG="logs/dsnot_task${TASK_ID}_gpu${GPU_SLOT}.log"
                
                launch_job "$GPU_SLOT" \
                    $DATASET_PREF "$DATASET" \
                    $MODEL_PREF "$MODEL" \
                    $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                    $SPARSITY_PREFIX "$SPARSITY" \
                    $COMPRESSION_PREF "$COMPRESSION" \
                    $OUTPUT_CSV_PREF "$OUTPUT_CSV" \
                    > "$LOG" 2>&1
                
                TASK_ID=$((TASK_ID + 1))
            done
        done
    done
done

# Attendi il completamento di tutti i job in background rimanenti
echo "Waiting for all jobs to finish..."
wait
echo "All done."
nvidia-smi

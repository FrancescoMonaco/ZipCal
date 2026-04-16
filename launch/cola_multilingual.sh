#!/bin/bash
set -e

MAX_JOBS=4          # Numero di job paralleli (GPU 2-3 per cola_multilingual.sh)
GPU_OFFSET=0        # Le GPU 0-1 sono usate da eval_multilingual.sh
declare -a PIDS=()  # Tracciamento dei PID in background

acquire_gpu() {
    while true; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset PIDS[$i]
                echo "$i"
                return
            fi
        done
        sleep 5
    done
}

launch_job() {
    local GPU_ID=$1; shift
    local REAL_GPU=$((GPU_ID + GPU_OFFSET))
    CUDA_VISIBLE_DEVICES=$REAL_GPU python source/eval_cola.py "$@" &
    PIDS[$GPU_ID]=$!
}

DATASET_PREF="--datasets"
DATASETS=(
    "xnli_es"
    "xquad_es"
    "xnli_zh"
    "xwinograd_zh"
    "xcopa_zh"
    "xquad_zh"
)

EVAL_TASKS_PREF="--eval_tasks"
EVAL_TASKS=(
    "xnli_es"
    "xquad_es"
    "xnli_zh"
    "xwinograd_zh"
    "xcopa_zh"
    "xquad_zh"
    "global_mmlu_es"
    "global_mmlu_zh"
)

MODEL_PREF="--model"
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it")
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128)
COMPRESSION_PREF="--compression_type"
COMPRESSION_TYPES=("2ssp")
OUTPUT_CSV_PREF="--output_csv"
OUTPUT_CSV="results/cola_experiments_multilingual.csv"

mkdir -p logs

TASK_ID=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for NSAMPLES in "${NUM_SAMPLES[@]}"; do
            for COMPRESSION in "${COMPRESSION_TYPES[@]}"; do

                if [[ ${#PIDS[@]} -lt $MAX_JOBS ]]; then
                    for ((g=0; g<MAX_JOBS; g++)); do
                        if [[ -z "${PIDS[$g]}" ]] || ! kill -0 "${PIDS[$g]}" 2>/dev/null; then
                            GPU_SLOT=$g
                            break
                        fi
                    done
                else
                    GPU_SLOT=$(acquire_gpu)
                fi

                echo "================================================================"
                echo "TASK $TASK_ID -> GPU $GPU_SLOT"
                echo "Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES, Compression: $COMPRESSION"
                echo "================================================================"

                LOG="logs/cola_multilingual_task${TASK_ID}_gpu${GPU_SLOT}.log"

                launch_job "$GPU_SLOT" \
                    $DATASET_PREF "$DATASET" \
                    $MODEL_PREF "$MODEL" \
                    $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                    $SPARSITY_PREFIX "$SPARSITY" \
                    $COMPRESSION_PREF "$COMPRESSION" \
                    $OUTPUT_CSV_PREF "$OUTPUT_CSV" \
                    $EVAL_TASKS_PREF "${EVAL_TASKS[@]}" \
                    > "$LOG" 2>&1

                TASK_ID=$((TASK_ID + 1))
            done
        done
    done
done

echo "Waiting for all jobs to finish..."
wait
echo "All done."
nvidia-smi

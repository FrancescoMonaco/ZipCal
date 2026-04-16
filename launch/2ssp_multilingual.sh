#!/bin/bash
set -e

MAX_JOBS=2          # Number of parallel jobs (GPU slots)
GPU_OFFSET=2        # Starting GPU index
declare -a PIDS=()  # Track background PIDs

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
    CUDA_VISIBLE_DEVICES=$REAL_GPU python "$@" &
    PIDS[$GPU_ID]=$!
}

DATASET_PREF="--datasets"
DATASETS=(
    "oscar_es"
    "oscar_zh"
    "mc4_es"
    "mc4_zh"
    "xnli_es"
    "xquad_es"
    "xnli_zh"
    "xwinograd_zh"
    "xcopa_zh"
    "xquad_zh"
    "oscar_es oscar_zh mc4_es mc4_zh"
    "xnli_es xquad_es xnli_zh xwinograd_zh xcopa_zh xquad_zh"
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
NUM_SAMPLES=(128)
SPARSITY_PREFIX="--sparsity"
SPARSITIES=("0.25")
COMPRESSION_PREF="--compression_type"
COMPRESSION="2ssp"
PRUNING_PREFIX="--pruning_types"
PRUNING_TYPES=("unique_tokens" "words_dataset")
OUTPUT_CSV_PREF="--output_csv"
OUTPUT_CSV="results/2ssp_experiment_results_multilingual.csv"

mkdir -p logs

TASK_ID=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for NSAMPLES in "${NUM_SAMPLES[@]}"; do
            for SPARSITY in "${SPARSITIES[@]}"; do
                for P_TYPE in "${PRUNING_TYPES[@]}"; do

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
                    echo "Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES, Sparsity: $SPARSITY, Calibration: $P_TYPE"
                    echo "================================================================"

                    LOG="logs/2ssp_multilingual_task${TASK_ID}_gpu${GPU_SLOT}.log"

                    launch_job "$GPU_SLOT" \
                        source/run_experiment.py \
                        $DATASET_PREF $DATASET \
                        $MODEL_PREF "$MODEL" \
                        $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                        $SPARSITY_PREFIX "$SPARSITY" \
                        $COMPRESSION_PREF "$COMPRESSION" \
                        $PRUNING_PREFIX "$P_TYPE" \
                        $OUTPUT_CSV_PREF "$OUTPUT_CSV" \
                        $EVAL_TASKS_PREF "${EVAL_TASKS[@]}" \
                        > "$LOG" 2>&1

                    TASK_ID=$((TASK_ID + 1))
                done
            done
        done
    done
done

COLA_OUTPUT_CSV="results/2ssp_cola_experiment_results_multilingual.csv"

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for NSAMPLES in "${NUM_SAMPLES[@]}"; do
            for SPARSITY in "${SPARSITIES[@]}"; do

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
                echo "TASK $TASK_ID -> GPU $GPU_SLOT  [COLA + 2SSP]"
                echo "Model: $MODEL, Dataset: $DATASET, Samples: $NSAMPLES, Sparsity: $SPARSITY"
                echo "================================================================"

                LOG="logs/2ssp_cola_multilingual_task${TASK_ID}_gpu${GPU_SLOT}.log"

                launch_job "$GPU_SLOT" \
                    source/eval_cola.py \
                    $DATASET_PREF $DATASET \
                    $MODEL_PREF "$MODEL" \
                    $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                    $SPARSITY_PREFIX "$SPARSITY" \
                    $COMPRESSION_PREF "$COMPRESSION" \
                    --pruning_types cola \
                    $OUTPUT_CSV_PREF "$COLA_OUTPUT_CSV" \
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

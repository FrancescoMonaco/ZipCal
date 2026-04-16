#!/bin/bash
set -e

MAX_JOBS=4          # number of parallel jobs (= number of GPUs)
declare -a PIDS=()  # track background PIDs

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
    CUDA_VISIBLE_DEVICES=$GPU_ID python source/run_experiment.py "$@" &
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
MODELS=("google/gemma-2-9b-it" "meta-llama/Llama-3.1-8B-Instruct")
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128)
COMPRESSION_PREFIX="--compression_type"
COMPRESSION_TYPE=("awq")
PRUNING_PREFIX="--pruning_types"
PRUNING_TYPES=("unique_tokens" "random" "most_similar" "least_perplexity" "distribution_matching" "words_dataset")

mkdir -p logs

TASK_ID=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for P_TYPE in "${PRUNING_TYPES[@]}"; do
            SUBTASK_ID=0
            for COMP in "${COMPRESSION_TYPE[@]}"; do
                for NSAMPLES in "${NUM_SAMPLES[@]}"; do
                    SUBTASK_ID=$((SUBTASK_ID + 1))

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
                    echo "TASK $TASK_ID | SUBTASK $SUBTASK_ID -> GPU $GPU_SLOT"
                    echo "Model=$MODEL, Dataset=$DATASET, Comp=$COMP, NSamples=$NSAMPLES, Pruning=$P_TYPE"
                    echo "================================================================"

                    LOG="logs/eval_awq_multilingual${TASK_ID}_s${SUBTASK_ID}_gpu${GPU_SLOT}.log"
                    launch_job "$GPU_SLOT" \
                        $DATASET_PREF $DATASET \
                        $MODEL_PREF "$MODEL" \
                        $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                        $SPARSITY_PREFIX "$SPARSITY" \
                        $COMPRESSION_PREFIX "$COMP" \
                        $PRUNING_PREFIX "$P_TYPE" \
                        $EVAL_TASKS_PREF "${EVAL_TASKS[@]}" \
                        > "$LOG" 2>&1

                done
            done
            TASK_ID=$((TASK_ID + 1))
        done
    done
done

echo "Waiting for all jobs to finish..."
wait
echo "All done."

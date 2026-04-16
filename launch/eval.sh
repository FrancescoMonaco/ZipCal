#!/bin/bash
set -e

TOTAL_GPUS=4        # physical GPUs visible on the node
GPUS_PER_JOB=2      # assign two GPUs to each run_experiment process
MAX_JOBS=$((TOTAL_GPUS / GPUS_PER_JOB))
declare -a PIDS=()  # track background PIDs

# Return the first free slot index, or -1 if all are busy
find_free_slot() {
    for ((i=0; i<MAX_JOBS; i++)); do
        if [[ -z "${PIDS[$i]}" ]] || ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "$i"
            return
        fi
    done
    echo "-1"
}

# Wait until a GPU slot is free, return the free GPU index
acquire_gpu() {
    while true; do
        local slot
        slot=$(find_free_slot)
        if [[ "$slot" != "-1" ]]; then
            echo "$slot"
            return
        fi
        # All slots busy - wait a moment and retry
        sleep 5
    done
}

# Launch a job on a specific GPU slot, background it
launch_job() {
    local SLOT_ID=$1; shift
    local START_GPU=$((SLOT_ID * GPUS_PER_JOB))
    local END_GPU=$((START_GPU + GPUS_PER_JOB - 1))
    local DEVICES
    DEVICES=$(seq -s, "$START_GPU" "$END_GPU")

    CUDA_VISIBLE_DEVICES=$DEVICES python source/run_experiment.py "$@" &
    PIDS[$SLOT_ID]=$!
}

DATASET_PREF="--datasets"
DATASETS=("winogrande" "arc_challenge" "boolq" "hellaswag" "openbookqa" "rte" "mmlu" "wmt14" "anli_r1" "svamp" "gsm8k" "pile" "wikitext" "c4" "winogrande arc_challenge boolq hellaswag openbookqa rte")
MODEL_PREF="--model"
MODELS=("meta-llama/Llama-3.1-70B-Instruct" ) #"google/gemma-2-9b-it" "meta-llama/Llama-3.1-8B-Instruct")
NUM_SAMPLES_PREFIX="--nsamples"
SPARSITY_PREFIX="--sparsity"
SPARSITY="0.25"
NUM_SAMPLES=(128)
COMPRESSION_PREFIX="--compression_type"
COMPRESSION_TYPE=("pruning" "quantization" "awq" "2ssp")
PRUNING_PREFIX="--pruning_types"
PRUNING_TYPES=("words_dataset") #"unique_tokens" "random" "most_similar" "least_perplexity" "distribution_matching" 

mkdir -p logs

TASK_ID=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for P_TYPE in "${PRUNING_TYPES[@]}"; do
            SUBTASK_ID=0
            for COMP in "${COMPRESSION_TYPE[@]}"; do
                for NSAMPLES in "${NUM_SAMPLES[@]}"; do
                    SUBTASK_ID=$((SUBTASK_ID + 1))

                    GPU_SLOT=$(find_free_slot)
                    if [[ "$GPU_SLOT" == "-1" ]]; then
                        GPU_SLOT=$(acquire_gpu)
                    fi

                    START_GPU=$((GPU_SLOT * GPUS_PER_JOB))
                    END_GPU=$((START_GPU + GPUS_PER_JOB - 1))
                    GPU_SET=$(seq -s, "$START_GPU" "$END_GPU")

                    echo "================================================================"
                    echo "TASK $TASK_ID | SUBTASK $SUBTASK_ID -> GPU slot $GPU_SLOT (CUDA_VISIBLE_DEVICES=$GPU_SET)"
                    echo "Model=$MODEL, Dataset=$DATASET, Comp=$COMP, NSamples=$NSAMPLES, Pruning=$P_TYPE"
                    echo "================================================================"

                    LOG="logs/eval_t${TASK_ID}_s${SUBTASK_ID}_gpu${GPU_SLOT}.log"
                    launch_job "$GPU_SLOT" \
                        $DATASET_PREF $DATASET \
                        $MODEL_PREF "$MODEL" \
                        $NUM_SAMPLES_PREFIX "$NSAMPLES" \
                        $SPARSITY_PREFIX "$SPARSITY" \
                        $COMPRESSION_PREFIX "$COMP" \
                        $PRUNING_PREFIX "$P_TYPE" \
                        > "$LOG" 2>&1

                done
            done
            TASK_ID=$((TASK_ID + 1))
        done
    done
done

# Wait for all remaining background jobs to finish
echo "Waiting for all jobs to finish..."
wait
echo "All done."

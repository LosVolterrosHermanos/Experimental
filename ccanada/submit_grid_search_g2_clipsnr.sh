#!/bin/bash

# Sbatch job submission script for tanea g2 and clipsnr grid search
# This script submits individual jobs for each parameter combination

# Fixed parameters
TANEA_KAPPA=0.75
WEIGHT_DECAY_TS=100
TRAIN_STEPS=90000
DECAY=""
SEQ_LEN=1024
BATCH_SIZE=32
VAL_BATCH_SIZE=32
VAL_STEPS=8
CLIP_NORM=100.0

# Fixed parameters for g2 sweep
TANEA_G3=0
MOMENTUM_FLAVOR="mk3"

# Arrays for grid search parameters
TANEA_G2_VALUES=(4E-4 3E-4 2E-4 1E-4)
CLIPSNR_VALUES=(32E-1 24E-1 16E-1 8E-1)

# Create results directory with timestamp in ccanada
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="grid_search_g2_clipsnr_results_${timestamp}"
mkdir -p "$results_dir"

# Log file for the grid search
log_file="$results_dir/grid_search_g2_clipsnr_submissions.log"

echo "Grid search job submission started at $(date)" | tee -a "$log_file"
echo "Results will be saved to: $results_dir" | tee -a "$log_file"
echo "Parameters:" | tee -a "$log_file"
echo "  tanea_g2: ${TANEA_G2_VALUES[*]}" | tee -a "$log_file"
echo "  clipsnr: ${CLIPSNR_VALUES[*]}" | tee -a "$log_file"
echo "  tanea_g3: $TANEA_G3 (fixed)" | tee -a "$log_file"
echo "  momentum_flavor: $MOMENTUM_FLAVOR (fixed)" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Counter for job submission
job_counter=0

# Submit Adam baseline jobs
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Adam baseline beta1=0.9, lr=3e-4" | tee -a "$log_file"

sbatch --job-name="adam_g2_b1_0.9" \
       --output="$results_dir/adam_g2_b1_0.9_%j.out" \
       --error="$results_dir/adam_g2_b1_0.9_%j.err" \
       --time=24:00:00 \
       --mem=32G \
       --cpus-per-task=2 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_adamw_baseline_mixed_bf16_rope.py \\
    --train_steps=$TRAIN_STEPS \\
    --batch_size=$BATCH_SIZE \\
    --val_batch_size=$VAL_BATCH_SIZE \\
    --val_steps=$VAL_STEPS \\
    --seq_len=$SEQ_LEN \\
    --lr=3E-4 \\
    --beta1=0.9 \\
    --beta2=0.95 \\
    --grad_clip=$CLIP_NORM \\
    --weight_decay=1E-3 \\
    --attention_implementation=naive \\
    --results_dir $results_dir
"

job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Adam baseline beta1=0.0, lr=8e-5" | tee -a "$log_file"

sbatch --job-name="adam_g2_b1_0.0" \
       --output="$results_dir/adam_g2_b1_0.0_%j.out" \
       --error="$results_dir/adam_g2_b1_0.0_%j.err" \
       --time=24:00:00 \
       --mem=32G \
       --cpus-per-task=2 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_adamw_baseline_mixed_bf16_rope.py \\
    --train_steps=$TRAIN_STEPS \\
    --batch_size=$BATCH_SIZE \\
    --val_batch_size=$VAL_BATCH_SIZE \\
    --val_steps=$VAL_STEPS \\
    --seq_len=$SEQ_LEN \\
    --lr=3E-4 \\
    --beta1=0.0 \\
    --beta2=0.95 \\
    --grad_clip=$CLIP_NORM \\
    --weight_decay=1E-3 \\
    --attention_implementation=naive \\
    --results_dir $results_dir
"

# Submit Tanea grid search jobs
for tanea_g2 in "${TANEA_G2_VALUES[@]}"; do
    for clipsnr in "${CLIPSNR_VALUES[@]}"; do
        job_counter=$((job_counter + 1))
        
        echo "Submitting job $job_counter: tanea_g2=$tanea_g2, clipsnr=$clipsnr" | tee -a "$log_file"
        
        sbatch --job-name="tanea_g2_${tanea_g2}_clipsnr_${clipsnr}" \
               --output="$results_dir/tanea_g2_${tanea_g2}_clipsnr_${clipsnr}_%j.out" \
               --error="$results_dir/tanea_g2_${tanea_g2}_clipsnr_${clipsnr}_%j.err" \
               --time=24:00:00 \
               --mem=32G \
               --cpus-per-task=2 \
               --gres=gpu:1 \
               --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_tanea_tau_stats_mixed_bf16_rope.py \\
    --train_steps=$TRAIN_STEPS \\
    --batch_size=$BATCH_SIZE \\
    --val_batch_size=$VAL_BATCH_SIZE \\
    --val_steps=$VAL_STEPS \\
    --seq_len=$SEQ_LEN \\
    --tanea_g2=$tanea_g2 \\
    --tanea_g3=$TANEA_G3 \\
    --tanea_kappa=$TANEA_KAPPA \\
    --clipsnr=$clipsnr \\
    --grad_clip=$CLIP_NORM \\
    --weight_decay=1E-3 \\
    --power_weight_decay=1.0 \\
    --weight_decay_ts=$WEIGHT_DECAY_TS \\
    --momentum_flavor $MOMENTUM_FLAVOR \\
    $DECAY \\
    --attention_implementation=naive \\
    --disable_checkpoint \\
    --results_dir $results_dir
"
    done
done

echo "=== Job submission completed ===" | tee -a "$log_file"
echo "Submitted $job_counter jobs at $(date)" | tee -a "$log_file"
echo "Results directory: $results_dir" | tee -a "$log_file"
echo "" | tee -a "$log_file"
echo "Check job status with: squeue -u \$USER" | tee -a "$log_file"
echo "Cancel all jobs with: scancel -u \$USER" | tee -a "$log_file"

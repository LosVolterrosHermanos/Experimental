#!/bin/bash

# Sbatch job submission script for tanea g3 and momentum flavor grid search
# This script submits individual jobs for each parameter combination

# Fixed parameters (using best values from previous sweep)
TANEA_KAPPA=0.75
WEIGHT_DECAY_TS=100
TRAIN_STEPS=120000
DECAY=""
SEQ_LEN=1024
BATCH_SIZE=8
VAL_BATCH_SIZE=8
VAL_STEPS=8
CLIP_NORM=100.0

# Fixed parameters from previous sweep
TANEA_G2=1E-4
CLIPSNR=2.0

# Arrays for grid search parameters
TANEA_G3_VALUES=(16E-5 8E-5 4E-5)
MOMENTUM_FLAVOR_VALUES=("effective-clip" "mk3")

# Momentum flavor scalers
MK2_SCALER=$(echo "scale=10; 11480/9332" | bc -l)
MK3_SCALER=$(echo "scale=10; 11480/7198" | bc -l)

# Create results directory with timestamp in ccanada
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="grid_search_g3_momentum_results_${timestamp}"
mkdir -p "$results_dir"

# Log file for the grid search
log_file="$results_dir/grid_search_g3_momentum_submissions.log"

echo "Grid search job submission started at $(date)" | tee -a "$log_file"
echo "Results will be saved to: $results_dir" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Counter for job submission
job_counter=0

# Submit Adam baseline jobs
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Adam baseline beta1=0.9" | tee -a "$log_file"

sbatch --job-name="adam_b1_0.9" \
       --output="$results_dir/adam_b1_0.9_%j.out" \
       --error="$results_dir/adam_b1_0.9_%j.err" \
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
    --lr=1E-4 \\
    --beta1=0.9 \\
    --beta2=0.95 \\
    --grad_clip=$CLIP_NORM \\
    --weight_decay=1E-3 \\
    --attention_implementation=naive \\
    --results_dir $results_dir
"

job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Adam baseline beta1=0.0" | tee -a "$log_file"

sbatch --job-name="adam_b1_0.0" \
       --output="$results_dir/adam_b1_0.0_%j.out" \
       --error="$results_dir/adam_b1_0.0_%j.err" \
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
    --lr=1E-4 \\
    --beta1=0.0 \\
    --beta2=0.95 \\
    --grad_clip=$CLIP_NORM \\
    --weight_decay=1E-3 \\
    --attention_implementation=naive \\
    --results_dir $results_dir
"

# Submit Tanea grid search jobs
for tanea_g3 in "${TANEA_G3_VALUES[@]}"; do
    for momentum_flavor in "${MOMENTUM_FLAVOR_VALUES[@]}"; do
        job_counter=$((job_counter + 1))
        
        # Apply momentum flavor scaling to g3
        if [ "$momentum_flavor" = "mk2" ]; then
            scaled_g3=$(python3 -c "print(f'{float(\"$tanea_g3\") * $MK2_SCALER:.3e}')")
        elif [ "$momentum_flavor" = "mk3" ]; then
            scaled_g3=$(python3 -c "print(f'{float(\"$tanea_g3\") * $MK3_SCALER:.3e}')")
        else
            scaled_g3=$tanea_g3
        fi
        
        echo "Submitting job $job_counter: tanea_g3=$tanea_g3 (scaled: $scaled_g3), momentum_flavor=$momentum_flavor" | tee -a "$log_file"
        
        sbatch --job-name="tanea_g3_${tanea_g3}_${momentum_flavor}" \
               --output="$results_dir/tanea_g3_${tanea_g3}_${momentum_flavor}_%j.out" \
               --error="$results_dir/tanea_g3_${tanea_g3}_${momentum_flavor}_%j.err" \
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
    --tanea_g2=$TANEA_G2 \\
    --tanea_g3=$scaled_g3 \\
    --tanea_kappa=$TANEA_KAPPA \\
    --clipsnr=$CLIPSNR \\
    --grad_clip=$CLIP_NORM \\
    --weight_decay=1E-3 \\
    --power_weight_decay=1.0 \\
    --weight_decay_ts=$WEIGHT_DECAY_TS \\
    --momentum_flavor $momentum_flavor \\
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

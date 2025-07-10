#!/bin/bash

# Sbatch job submission script for nanogpt speedrun comparison
# This script submits jobs for AdamW baseline and Tanea variants

# Default parameter values from timescale-experiment baseline files
# AdamW parameters from nanogpt_adamw_baseline_mixed_bf16_rope.py
DEFAULT_ADAMW_LR="16E-5"
DEFAULT_ADAMW_BETA1="0.9"
DEFAULT_ADAMW_BETA2="0.95"
DEFAULT_ADAMW_WEIGHT_DECAY="1E-3"

# Tanea parameters from nanogpt_tanea_tau_stats_mixed_bf16_rope.py  
DEFAULT_TANEA_G2="16E-5"
DEFAULT_TANEA_G3="4E-5"
DEFAULT_TANEA_CLIPSNR="2.0"

# Create results directory with timestamp in ccanada
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="nanogpt_speedrun_results_${timestamp}"
mkdir -p "$results_dir"

# Log file for the speedrun
log_file="$results_dir/nanogpt_speedrun_submissions.log"

echo "NanoGPT speedrun job submission started at $(date)" | tee -a "$log_file"
echo "Results will be saved to: $results_dir" | tee -a "$log_file"
echo "Default parameters:" | tee -a "$log_file"
echo "  AdamW LR: $DEFAULT_ADAMW_LR, Beta1: $DEFAULT_ADAMW_BETA1, Beta2: $DEFAULT_ADAMW_BETA2, WD: $DEFAULT_ADAMW_WEIGHT_DECAY" | tee -a "$log_file"
echo "  Tanea G2: $DEFAULT_TANEA_G2, G3: $DEFAULT_TANEA_G3, ClipSNR: $DEFAULT_TANEA_CLIPSNR" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Counter for job submission
job_counter=0

# Submit AdamW baseline job with lr=24E-5, weight_decay=0, --enable_wsd
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: AdamW baseline with lr=24E-5, weight_decay=0, WSD enabled" | tee -a "$log_file"

sbatch --job-name="adamw_wsd_baseline" \
       --output="$results_dir/adamw_wsd_baseline_%j.out" \
       --error="$results_dir/adamw_wsd_baseline_%j.err" \
       --time=24:00:00 \
       --mem=16G \
       --cpus-per-task=1 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_adamw_baseline_mixed_bf16_rope.py \\
    --lr 24E-5 \\
    --beta1 $DEFAULT_ADAMW_BETA1 \\
    --beta2 $DEFAULT_ADAMW_BETA2 \\
    --weight_decay 0 \\
    --enable_wsd \\
    --results_dir $results_dir
"

# Submit AdamW baseline job with lr=48E-5 and --enable_wsd
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: AdamW baseline with lr=48E-5 and WSD" | tee -a "$log_file"

sbatch --job-name="adamw_2x_lr_wsd_baseline" \
       --output="$results_dir/adamw_2x_lr_wsd_baseline_%j.out" \
       --error="$results_dir/adamw_2x_lr_wsd_baseline_%j.err" \
       --time=24:00:00 \
       --mem=16G \
       --cpus-per-task=1 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_adamw_baseline_mixed_bf16_rope.py \\
    --lr 48E-5 \\
    --beta1 $DEFAULT_ADAMW_BETA1 \\
    --beta2 $DEFAULT_ADAMW_BETA2 \\
    --weight_decay $DEFAULT_ADAMW_WEIGHT_DECAY \\
    --enable_wsd \\
    --grad_clip 2.0 \\
    --warmup_fraction 0.1 \\
    --results_dir $results_dir
"

# Submit AdamW baseline job with lr=48E-5, grad_clip=1.0, and --enable_wsd
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: AdamW baseline with lr=48E-5, grad_clip=1.0, and WSD" | tee -a "$log_file"

sbatch --job-name="adamw_4x_lr_ext_warmup_wsd" \
       --output="$results_dir/adamw_4x_lr_ext_warmup_wsd_%j.out" \
       --error="$results_dir/adamw_4x_lr_ext_warmup_wsd_%j.err" \
       --time=24:00:00 \
       --mem=16G \
       --cpus-per-task=1 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_adamw_baseline_mixed_bf16_rope.py \\
    --lr 48E-5 \\
    --beta1 $DEFAULT_ADAMW_BETA1 \\
    --beta2 $DEFAULT_ADAMW_BETA2 \\
    --weight_decay $DEFAULT_ADAMW_WEIGHT_DECAY \\
    --enable_wsd \\
    --grad_clip 1.0 \\
    --warmup_fraction 0.1 \\
    --results_dir $results_dir
"

# Submit Tanea job 1: tanea_g2=32E-5, g3=32E-5, kappa=1.0, clipsnr=1.6, mk3 momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Tanea g2=32E-5, g3=32E-5, kappa=1.0, clipsnr=1.6, mk3 momentum, WSD enabled" | tee -a "$log_file"

sbatch --job-name="tanea_g2_32E-5_g3_32E-5_kappa_1.0_mk3_wsd" \
       --output="$results_dir/tanea_g2_32E-5_g3_32E-5_kappa_1.0_mk3_wsd_%j.out" \
       --error="$results_dir/tanea_g2_32E-5_g3_32E-5_kappa_1.0_mk3_wsd_%j.err" \
       --time=24:00:00 \
       --mem=16G \
       --cpus-per-task=1 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_tanea_tau_stats_mixed_bf16_rope.py \\
    --tanea_g2=32E-5 \\
    --tanea_g3=32E-5 \\
    --tanea_kappa=1.0 \\
    --clipsnr=1.6 \\
    --momentum_flavor=mk3 \\
    --enable_wsd \\
    --disable_checkpoint \\
    --results_dir $results_dir
"

# Submit Tanea job 2: tanea_g2=32E-5, g3=8E-5, kappa=0.5, clipsnr=1.6, mk3 momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Tanea g2=32E-5, g3=8E-5, kappa=0.5, clipsnr=1.6, mk3 momentum, WSD enabled" | tee -a "$log_file"

sbatch --job-name="tanea_g2_32E-5_g3_8E-5_kappa_0.5_mk3_wsd" \
       --output="$results_dir/tanea_g2_32E-5_g3_8E-5_kappa_0.5_mk3_wsd_%j.out" \
       --error="$results_dir/tanea_g2_32E-5_g3_8E-5_kappa_0.5_mk3_wsd_%j.err" \
       --time=24:00:00 \
       --mem=16G \
       --cpus-per-task=1 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_tanea_tau_stats_mixed_bf16_rope.py \\
    --tanea_g2=32E-5 \\
    --tanea_g3=8E-5 \\
    --tanea_kappa=0.5 \\
    --clipsnr=1.6 \\
    --momentum_flavor=mk3 \\
    --enable_wsd \\
    --disable_checkpoint \\
    --results_dir $results_dir
"

# Submit Tanea job 3: tanea_g2=32E-5, g3=32E-5, kappa=0.65, clipsnr=1.6, mk3 momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Tanea g2=32E-5, g3=32E-5, kappa=0.65, clipsnr=1.6, mk3 momentum, WSD enabled" | tee -a "$log_file"

sbatch --job-name="tanea_g2_32E-5_g3_32E-5_kappa_0.65_mk3_wsd" \
       --output="$results_dir/tanea_g2_32E-5_g3_32E-5_kappa_0.65_mk3_wsd_%j.out" \
       --error="$results_dir/tanea_g2_32E-5_g3_32E-5_kappa_0.65_mk3_wsd_%j.err" \
       --time=24:00:00 \
       --mem=16G \
       --cpus-per-task=1 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_tanea_tau_stats_mixed_bf16_rope.py \\
    --tanea_g2=32E-5 \\
    --tanea_g3=32E-5 \\
    --tanea_kappa=0.65 \\
    --clipsnr=1.6 \\
    --momentum_flavor=mk3 \\
    --enable_wsd \\
    --disable_checkpoint \\
    --results_dir $results_dir
"

# Submit Tanea job 4: tanea_g2=32E-5, g3=64E-5, kappa=0.75, clipsnr=1.6, mk3 momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Tanea g2=32E-5, g3=64E-5, kappa=0.75, clipsnr=1.6, mk3 momentum, WSD enabled" | tee -a "$log_file"

sbatch --job-name="tanea_g2_32E-5_g3_64E-5_kappa_0.75_mk3_wsd" \
       --output="$results_dir/tanea_g2_32E-5_g3_64E-5_kappa_0.75_mk3_wsd_%j.out" \
       --error="$results_dir/tanea_g2_32E-5_g3_64E-5_kappa_0.75_mk3_wsd_%j.err" \
       --time=24:00:00 \
       --mem=16G \
       --cpus-per-task=1 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_tanea_tau_stats_mixed_bf16_rope.py \\
    --tanea_g2=32E-5 \\
    --tanea_g3=64E-5 \\
    --tanea_kappa=0.75 \\
    --clipsnr=1.6 \\
    --momentum_flavor=mk3 \\
    --enable_wsd \\
    --disable_checkpoint \\
    --results_dir $results_dir
"

# Submit Tanea job 5: tanea_g2=32E-5, g3=128E-5, kappa=0.75, clipsnr=1.6, mk3 momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter: Tanea g2=32E-5, g3=128E-5, kappa=0.75, clipsnr=1.6, mk3 momentum, WSD enabled" | tee -a "$log_file"

sbatch --job-name="tanea_g2_32E-5_g3_128E-5_kappa_0.75_mk3_wsd" \
       --output="$results_dir/tanea_g2_32E-5_g3_128E-5_kappa_0.75_mk3_wsd_%j.out" \
       --error="$results_dir/tanea_g2_32E-5_g3_128E-5_kappa_0.75_mk3_wsd_%j.err" \
       --time=24:00:00 \
       --mem=16G \
       --cpus-per-task=1 \
       --gres=gpu:1 \
       --wrap="
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
cd /home/epaq/Experimental/timescale-experiment
python nanogpt_tanea_tau_stats_mixed_bf16_rope.py \\
    --tanea_g2=32E-5 \\
    --tanea_g3=128E-5 \\
    --tanea_kappa=0.75 \\
    --clipsnr=1.6 \\
    --momentum_flavor=mk3 \\
    --enable_wsd \\
    --disable_checkpoint \\
    --results_dir $results_dir
"


echo "=== Job submission completed ===" | tee -a "$log_file"
echo "Submitted $job_counter jobs at $(date)" | tee -a "$log_file"
echo "Results directory: $results_dir" | tee -a "$log_file"
echo "" | tee -a "$log_file"
echo "Check job status with: squeue -u \$USER" | tee -a "$log_file"
echo "Cancel all jobs with: scancel -u \$USER" | tee -a "$log_file"

# Function to parse hyperparameters from error files
parse_hyperparameters() {
    local results_dir="$1"
    echo "=== Parsing hyperparameters from error files ===" | tee -a "$results_dir/nanogpt_speedrun_submissions.log"
    
    for err_file in "$results_dir"/*.err; do
        if [[ -f "$err_file" ]]; then
            local basename=$(basename "$err_file" .err)
            echo "Processing: $basename" | tee -a "$results_dir/nanogpt_speedrun_submissions.log"
            
            # Parse Tanea parameters
            local tanea_params=$(grep -E "INFO - Tanea params:" "$err_file" | head -1)
            if [[ -n "$tanea_params" ]]; then
                local g2=$(echo "$tanea_params" | grep -oE "g2=[0-9.e-]+" | cut -d'=' -f2)
                local g3=$(echo "$tanea_params" | grep -oE "g3=[0-9.e-]+" | cut -d'=' -f2)
                local kappa=$(echo "$tanea_params" | grep -oE "kappa=[0-9.e-]+" | cut -d'=' -f2)
                echo "  Tanea - g2: $g2, g3: $g3, kappa: $kappa" | tee -a "$results_dir/nanogpt_speedrun_submissions.log"
            fi
            
            # Parse AdamW parameters
            local adamw_params=$(grep -E "LR: [0-9.e-]+, Beta1: [0-9.e-]+, Beta2: [0-9.e-]+, WD: [0-9.e-]+" "$err_file" | head -1)
            if [[ -n "$adamw_params" ]]; then
                local lr=$(echo "$adamw_params" | grep -oE "LR: [0-9.e-]+" | cut -d' ' -f2)
                local beta1=$(echo "$adamw_params" | grep -oE "Beta1: [0-9.e-]+" | cut -d' ' -f2)
                local beta2=$(echo "$adamw_params" | grep -oE "Beta2: [0-9.e-]+" | cut -d' ' -f2)
                local wd=$(echo "$adamw_params" | grep -oE "WD: [0-9.e-]+" | cut -d' ' -f2)
                echo "  AdamW - LR: $lr, Beta1: $beta1, Beta2: $beta2, WD: $wd" | tee -a "$results_dir/nanogpt_speedrun_submissions.log"
            fi
        fi
    done
    
    echo "=== Hyperparameter parsing completed ===" | tee -a "$results_dir/nanogpt_speedrun_submissions.log"
}

# Usage: parse_hyperparameters <results_directory>
# Example: parse_hyperparameters nanogpt_speedrun_results_20250710_075036

#!/bin/bash

# Sbatch job submission script for nanogpt speedrun comparison
# This script submits jobs for AdamW baseline and Tanea variants

DATA_ROOT="/home/c/cypaquet/Experimental/dana-nonquadratic-tests/gpt2/fineweb-edu/sample/100BT"

# Default parameter values from timescale-experiment baseline files
# AdamW parameters from nanogpt_adamw_baseline_mixed_bf16_rope.py
DEFAULT_ADAMW_LR="16E-5"
DEFAULT_ADAMW_BETA1="0.9"
DEFAULT_ADAMW_BETA2="0.95"

# Tanea parameters from nanogpt_tanea_tau_stats_mixed_bf16_rope.py  
DEFAULT_TANEA_G2="16E-5"
DEFAULT_TANEA_G3="4E-5"
DEFAULT_TANEA_CLIPSNR="1.6"
GRAD_CLIP="100"

# Common sbatch options
COMMON_TIME="24:00:00"
COMMON_MEM="384G"
COMMON_CPUS="2"
COMMON_NODES="1"
COMMON_NTASKS="1"
COMMON_GRES="gpu:4"

# Common python training options
COMMON_MODEL_SIZE="GPT2-nano"
COMMON_ATTENTION_IMPL="naive"
COMMON_BATCH_SIZE="32"
COMMON_TRAIN_STEPS="90000"
COMMON_VAL_BATCH_SIZE="32"
COMMON_VAL_STEPS="20"

# Create results directory with timestamp in ccanada
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="nanogpt_speedrun_results_${timestamp}"
mkdir -p "$results_dir"

# Log file for the speedrun
log_file="$results_dir/nanogpt_speedrun_submissions.log"

echo "NanoGPT speedrun job submission started at $(date)" | tee -a "$log_file"
echo "Results will be saved to: $results_dir" | tee -a "$log_file"
echo "Default parameters:" | tee -a "$log_file"
echo "  AdamW LR: $DEFAULT_ADAMW_LR, Beta1: $DEFAULT_ADAMW_BETA1, Beta2: $DEFAULT_ADAMW_BETA2" | tee -a "$log_file"
echo "  Tanea G2: $DEFAULT_TANEA_G2, G3: $DEFAULT_TANEA_G3, ClipSNR: $DEFAULT_TANEA_CLIPSNR" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Function to create modified nodesetup.sh and submit job
submit_job() {
    local job_name="$1"
    local python_script="$2"
    local python_args="$3"
    
    # Create modified nodesetup.sh
    local nodesetup_file="$results_dir/nodesetup_${job_name}.sh"
    
    cat > "$nodesetup_file" << EOF
#!/bin/bash
#SBATCH --time=${COMMON_TIME}
#SBATCH --mem=${COMMON_MEM}
#SBATCH --cpus-per-task=${COMMON_CPUS}
#SBATCH --nodes=${COMMON_NODES}
#SBATCH --ntasks-per-node=${COMMON_NTASKS}
#SBATCH --gres=${COMMON_GRES}

module load python/3.13.2
module load cuda/12.6
module load gcc
module load arrow/19.0.1

#virtualenv --no-download /home/c/cypaquet/jaxenv
source /home/c/cypaquet/jaxenv/bin/activate
#pip install --no-index jax[cuda12] flax pandas optax matplotlib tiktoken huggingface-hub

echo "Loaded everything."

export TIKTOKEN_CACHE_DIR=/home/c/cypaquet/Experimental/tiktoken_cache

cd /home/c/cypaquet/Experimental/timescale-experiment

python $python_script $python_args
EOF
    
    # Make the file executable
    chmod +x "$nodesetup_file"
    
    # Submit the job
    sbatch --job-name="$job_name" \
           --output="$results_dir/${job_name}_%j.out" \
           --error="$results_dir/${job_name}_%j.err" \
           "$nodesetup_file"
}

# Counter for job submission
job_counter=0

# Submit AdamW baseline job with lr=16E-5, weight_decay=1E-3, --enable_wsd
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter:" | tee -a "$log_file"

submit_job "adam_$job_counter" "nanogpt_adamw_baseline.py" "\
    --model_size $COMMON_MODEL_SIZE \
    --attention_implementation $COMMON_ATTENTION_IMPL \
    --data_root $DATA_ROOT \
    --batch_size $COMMON_BATCH_SIZE \
    --train_steps $COMMON_TRAIN_STEPS \
    --val_batch_size $COMMON_VAL_BATCH_SIZE \
    --val_steps $COMMON_VAL_STEPS \
    --lr 16E-5 \
    --beta1 $DEFAULT_ADAMW_BETA1 \
    --beta2 $DEFAULT_ADAMW_BETA2 \
    --grad_clip $GRAD_CLIP \
    --weight_decay 1E-3 \
    --enable_wsd \
    --warmup_fraction 0.02 \
    --decay_fraction 0.2 \
    --results_dir /home/c/cypaquet/Experimental/ccanada/$results_dir"

# Submit Tanea job 1: tanea_g2=16E-5, g3=1E-4, kappa=0.75, clipsnr=1.6, always-on-mk2 momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter:" | tee -a "$log_file"

submit_job "tanea_$job_counter" "nanogpt_tanea.py" "\
    --model_size $COMMON_MODEL_SIZE \
    --attention_implementation $COMMON_ATTENTION_IMPL \
    --data_root $DATA_ROOT \
    --batch_size $COMMON_BATCH_SIZE \
    --train_steps $COMMON_TRAIN_STEPS \
    --val_batch_size $COMMON_VAL_BATCH_SIZE \
    --val_steps $COMMON_VAL_STEPS \
    --tanea_g2=16E-5 \
    --tanea_g3=1E-4 \
    --tanea_kappa=0.75 \
    --clipsnr=1.6 \
    --grad_clip $GRAD_CLIP \
    --momentum_flavor=always-on-mk2 \
    --enable_wsd \
    --warmup_fraction 0.02 \
    --decay_fraction 0.2 \
    --weight_decay 1E-3 \
    --weight_decay_ts 100 \
    --disable_checkpoint \
    --results_dir /home/c/cypaquet/Experimental/ccanada/$results_dir"

# Submit Tanea job 2: tanea_g2=16E-5, g3=1E-4, kappa=0.75, clipsnr=1.6, always-on momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter:" | tee -a "$log_file"

submit_job "tanea_$job_counter" "nanogpt_tanea.py" "\
    --model_size $COMMON_MODEL_SIZE \
    --attention_implementation $COMMON_ATTENTION_IMPL \
    --data_root $DATA_ROOT \
    --batch_size $COMMON_BATCH_SIZE \
    --train_steps $COMMON_TRAIN_STEPS \
    --val_batch_size $COMMON_VAL_BATCH_SIZE \
    --val_steps $COMMON_VAL_STEPS \
    --tanea_g2=16E-5 \
    --tanea_g3=1E-4 \
    --tanea_kappa=0.75 \
    --clipsnr=1.6 \
    --grad_clip $GRAD_CLIP \
    --momentum_flavor=always-on \
    --enable_wsd \
    --warmup_fraction 0.02 \
    --decay_fraction 0.2 \
    --weight_decay 1E-3 \
    --weight_decay_ts 100 \
    --disable_checkpoint \
    --results_dir /home/c/cypaquet/Experimental/ccanada/$results_dir"

# Submit Tanea job 3: tanea_g2=16E-5, g3=16E-5, kappa=0.75, clipsnr=1.6, mk3 momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter:" | tee -a "$log_file"

submit_job "tanea_$job_counter" "nanogpt_tanea.py" "\
    --data_root $DATA_ROOT \
    --model_size $COMMON_MODEL_SIZE \
    --attention_implementation $COMMON_ATTENTION_IMPL \
    --batch_size $COMMON_BATCH_SIZE \
    --train_steps $COMMON_TRAIN_STEPS \
    --val_batch_size $COMMON_VAL_BATCH_SIZE \
    --val_steps $COMMON_VAL_STEPS \
    --tanea_g2=16E-5 \
    --tanea_g3=16E-5 \
    --tanea_kappa=0.75 \
    --clipsnr=1.6 \
    --grad_clip $GRAD_CLIP \
    --momentum_flavor=mk3 \
    --enable_wsd \
    --warmup_fraction 0.02 \
    --decay_fraction 0.2 \
    --weight_decay 1E-3 \
    --weight_decay_ts 100 \
    --disable_checkpoint \
    --results_dir /home/c/cypaquet/Experimental/ccanada/$results_dir"

# Submit Tanea job 4: tanea_g2=16E-5, g3=16E-5, kappa=0.75, clipsnr=1.6, effective-clip momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter:" | tee -a "$log_file"

submit_job "tanea_$job_counter" "nanogpt_tanea.py" "\
    --data_root $DATA_ROOT \
    --model_size $COMMON_MODEL_SIZE \
    --attention_implementation $COMMON_ATTENTION_IMPL \
    --batch_size $COMMON_BATCH_SIZE \
    --train_steps $COMMON_TRAIN_STEPS \
    --val_batch_size $COMMON_VAL_BATCH_SIZE \
    --val_steps $COMMON_VAL_STEPS \
    --tanea_g2=16E-5 \
    --tanea_g3=16E-5 \
    --tanea_kappa=0.75 \
    --clipsnr=1.6 \
    --grad_clip $GRAD_CLIP \
    --momentum_flavor=effective-clip \
    --enable_wsd \
    --warmup_fraction 0.02 \
    --decay_fraction 0.2 \
    --weight_decay 1E-3 \
    --weight_decay_ts 100 \
    --disable_checkpoint \
    --results_dir /home/c/cypaquet/Experimental/ccanada/$results_dir"

# Submit Tanea job 5: tanea_g2=16E-5, g3=25.6E-5, kappa=1.0, clipsnr=1.6, mk3 momentum, --enable_wsd, --disable_checkpoint
job_counter=$((job_counter + 1))
echo "Submitting job $job_counter:" | tee -a "$log_file"

submit_job "tanea_$job_counter" "nanogpt_tanea.py" "\
    --data_root $DATA_ROOT \
    --model_size $COMMON_MODEL_SIZE \
    --attention_implementation $COMMON_ATTENTION_IMPL \
    --batch_size $COMMON_BATCH_SIZE \
    --train_steps $COMMON_TRAIN_STEPS \
    --val_batch_size $COMMON_VAL_BATCH_SIZE \
    --val_steps $COMMON_VAL_STEPS \
    --tanea_g2=16E-5 \
    --tanea_g3=25.6E-5 \
    --tanea_kappa=1.0 \
    --clipsnr=1.6 \
    --grad_clip $GRAD_CLIP \
    --momentum_flavor=mk3 \
    --enable_wsd \
    --warmup_fraction 0.02 \
    --decay_fraction 0.2 \
    --weight_decay 1E-3 \
    --weight_decay_ts 100 \
    --disable_checkpoint \
    --results_dir /home/c/cypaquet/Experimental/ccanada/$results_dir"

echo "=== Job submission completed ===" | tee -a "$log_file"
echo "Submitted $job_counter jobs at $(date)" | tee -a "$log_file"
echo "Results directory: $results_dir" | tee -a "$log_file"
echo "" | tee -a "$log_file"
echo "Check job status with: squeue -u \$USER" | tee -a "$log_file"
echo "Cancel all jobs with: scancel -u \$USER" | tee -a "$log_file"
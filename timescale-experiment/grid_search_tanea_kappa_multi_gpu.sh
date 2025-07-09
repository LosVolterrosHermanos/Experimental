#!/bin/bash

# Grid search script for tanea kappa parameter using multi-GPU training
# Parameters to vary:
# - tanea_kappa: {0.25, 0.5, 0.75, 1.0, 1.25, 1.5}
# - momentum_flavor: {effective-clip, mk2, mk3}
# Plus Adam baseline for comparison
# Uses sbatch for SLURM job submission

# Fixed parameters (using best values from previous sweeps)
TANEA_G2=4E-5
TANEA_G3=1E-5
TANEA_DELTA=8.0
WEIGHT_DECAY_TS=100
TRAIN_STEPS=120000
SEQ_LEN=1024
BATCH_SIZE=32  # 4x larger for 4 GPU system
VAL_BATCH_SIZE=32
VAL_STEPS=8
GRAD_CLIP=100.0
CLIPSNR=2.0

# Arrays for grid search parameters
TANEA_KAPPA_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)
MOMENTUM_FLAVOR_VALUES=("mk3")

# SLURM job configuration
PARTITION="gpu"          # Adjust for your cluster
NUM_GPUS=4
NUM_NODES=1
TIME_LIMIT="08:00:00"    # 8 hours per job
MEMORY="128G"             # Adjust based on your system
ACCOUNT=""               # Set your SLURM account if needed
QOS=""                   # Set your QOS if needed

# Counter for tracking progress
total_combinations=$(( ${#TANEA_KAPPA_VALUES[@]} * ${#MOMENTUM_FLAVOR_VALUES[@]} + 1 ))  # +1 for Adam baseline
current=0

echo "Starting grid search for tanea kappa parameter using multi-GPU training"
echo "Total combinations: $total_combinations"
echo "Parameters:"
echo "  tanea_kappa: ${TANEA_KAPPA_VALUES[*]}"
echo "  momentum_flavor: ${MOMENTUM_FLAVOR_VALUES[*]}"
echo "  tanea_g2: $TANEA_G2 (fixed)"
echo "  tanea_g3: $TANEA_G3 (fixed)"
echo "  Adam baseline: beta1=0.9, beta2=0.95, lr=4e-5, weight_decay=1e-3"
echo "  Fixed: tanea_delta=$TANEA_DELTA, weight_decay_ts=$WEIGHT_DECAY_TS, train_steps=$TRAIN_STEPS"
echo "  Multi-GPU: batch_size=$BATCH_SIZE (across $NUM_GPUS GPUs)"
echo ""

# Create results directory with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="grid_search_kappa_multi_gpu_results_${timestamp}"
mkdir -p "$results_dir"

# Log file for the grid search
log_file="$results_dir/grid_search_kappa_multi_gpu.log"

echo "Grid search started at $(date)" | tee -a "$log_file"
echo "Results will be saved to: $results_dir" | tee -a "$log_file"
echo "SLURM configuration: partition=$PARTITION, gpus=$NUM_GPUS, time=$TIME_LIMIT" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Function to create and submit SLURM job
submit_job() {
    local job_name="$1"
    local script_command="$2"
    local dependency="$3"
    
    # Create SLURM job script
    job_script="${results_dir}/${job_name}.sh"
    
    cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=$NUM_GPUS
#SBATCH --time=$TIME_LIMIT
#SBATCH --mem=$MEMORY
#SBATCH --output=${results_dir}/${job_name}_%j.out
#SBATCH --error=${results_dir}/${job_name}_%j.err
EOF

    # Add account and QOS if specified
    if [ -n "$ACCOUNT" ]; then
        echo "#SBATCH --account=$ACCOUNT" >> "$job_script"
    fi
    
    if [ -n "$QOS" ]; then
        echo "#SBATCH --qos=$QOS" >> "$job_script"
    fi
    
    # Add dependency if specified
    if [ -n "$dependency" ]; then
        echo "#SBATCH --dependency=afterok:$dependency" >> "$job_script"
    fi
    
    cat >> "$job_script" << EOF

# Load necessary modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment if needed
# source \$HOME/venv/bin/activate

# Set up environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Change to working directory
cd \$SLURM_SUBMIT_DIR

# Log job info
echo "Job started at: \$(date)"
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
echo "SLURM_JOB_NAME: \$SLURM_JOB_NAME"
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"
echo "Number of GPUs available: \$(nvidia-smi -L | wc -l)"
echo ""

# Run the command
$script_command

# Log completion
echo ""
echo "Job completed at: \$(date)"
EOF

    # Submit the job
    if [ -n "$dependency" ]; then
        job_id=$(sbatch --dependency=afterok:$dependency "$job_script" | awk '{print $4}')
    else
        job_id=$(sbatch "$job_script" | awk '{print $4}')
    fi
    
    echo "Submitted job $job_name with ID: $job_id" | tee -a "$log_file"
    echo "$job_id"
}

# Submit Adam baseline job first
current=$((current + 1))
echo "=== Submitting job $current/$total_combinations (Adam Baseline) ===" | tee -a "$log_file"

adam_command="python nanogpt_adamw_baseline_mixed_bf16_rope_multi_gpu.py \\
    --train_steps $TRAIN_STEPS \\
    --batch_size $BATCH_SIZE \\
    --val_batch_size $VAL_BATCH_SIZE \\
    --val_steps $VAL_STEPS \\
    --seq_len $SEQ_LEN \\
    --lr 4E-5 \\
    --beta1 0.9 \\
    --beta2 0.95 \\
    --grad_clip $GRAD_CLIP \\
    --weight_decay 1E-3 \\
    --attention_implementation xla \\
    --results_dir $results_dir"

adam_job_id=$(submit_job "adam_baseline" "$adam_command")

# Store job IDs for dependency management
job_ids=("$adam_job_id")

# Grid search loop for Tanea with kappa sweep
for tanea_kappa in "${TANEA_KAPPA_VALUES[@]}"; do
    for momentum_flavor in "${MOMENTUM_FLAVOR_VALUES[@]}"; do
        current=$((current + 1))
        
        job_name="tanea_kappa_${tanea_kappa}_${momentum_flavor}"
        
        echo "=== Submitting job $current/$total_combinations ===" | tee -a "$log_file"
        echo "Parameters: tanea_kappa=$tanea_kappa, momentum_flavor=$momentum_flavor" | tee -a "$log_file"
        
        tanea_command="python nanogpt_tanea_tau_stats_mixed_bf16_rope_multi_gpu.py \\
            --train_steps $TRAIN_STEPS \\
            --batch_size $BATCH_SIZE \\
            --val_batch_size $VAL_BATCH_SIZE \\
            --val_steps $VAL_STEPS \\
            --seq_len $SEQ_LEN \\
            --tanea_g2 $TANEA_G2 \\
            --tanea_g3 $TANEA_G3 \\
            --tanea_delta $TANEA_DELTA \\
            --tanea_kappa $tanea_kappa \\
            --clipsnr $CLIPSNR \\
            --grad_clip $GRAD_CLIP \\
            --weight_decay 1E-3 \\
            --power_weight_decay 1.0 \\
            --weight_decay_ts $WEIGHT_DECAY_TS \\
            --momentum_flavor $momentum_flavor \\
            --attention_implementation xla \\
            --disable_checkpoint \\
            --results_dir $results_dir"
        
        # Submit job (no dependency for parallel execution)
        job_id=$(submit_job "$job_name" "$tanea_command")
        job_ids+=("$job_id")
        
        echo "Submitted at: $(date)" | tee -a "$log_file"
        echo "" | tee -a "$log_file"
    done
done

# Create a final summary job that runs after all training jobs complete
summary_dependencies=$(IFS=:; echo "${job_ids[*]}")

summary_command="python plot_nanogpt_tanea_kappa_multi_gpu_visualization.py \\
    --results_dir $results_dir \\
    --pattern '*tanea_results*multi_gpu*.pkl' \\
    --adamw_pattern '*adamw_baseline*multi_gpu*.pkl' \\
    --output_prefix nanogpt_tanea_kappa_multi_gpu"

summary_job_id=$(submit_job "kappa_summary" "$summary_command" "$summary_dependencies")

echo "=== All jobs submitted ===" | tee -a "$log_file"
echo "Submitted at: $(date)" | tee -a "$log_file"
echo "Total jobs submitted: $current" | tee -a "$log_file"
echo "Results directory: $results_dir" | tee -a "$log_file"
echo "Summary job ID: $summary_job_id" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Create a job monitoring script
monitor_script="${results_dir}/monitor_jobs.sh"
cat > "$monitor_script" << EOF
#!/bin/bash
# Monitor script for kappa sweep jobs

echo "Monitoring jobs for kappa sweep (started $(date))"
echo "Results directory: $results_dir"
echo ""

# Job IDs
job_ids=(${job_ids[@]} $summary_job_id)

echo "Job IDs: \${job_ids[*]}"
echo ""

# Function to check job status
check_jobs() {
    echo "Job status at \$(date):"
    for job_id in "\${job_ids[@]}"; do
        status=\$(squeue -j \$job_id -h -o "%T" 2>/dev/null || echo "COMPLETED/NOT_FOUND")
        echo "  Job \$job_id: \$status"
    done
    echo ""
}

# Check status every 30 minutes
while true; do
    check_jobs
    
    # Check if all jobs are done
    running_jobs=\$(squeue -j \$(IFS=,; echo "\${job_ids[*]}") -h 2>/dev/null | wc -l)
    if [ \$running_jobs -eq 0 ]; then
        echo "All jobs completed at \$(date)"
        break
    fi
    
    echo "Waiting 30 minutes before next check..."
    sleep 1800  # 30 minutes
done

echo ""
echo "Grid search completed!"
echo "Results are in: $results_dir"
echo "Check the summary files for analysis results."
EOF

chmod +x "$monitor_script"

echo "Created job monitoring script: $monitor_script"
echo "Run it with: ./$monitor_script"
echo ""
echo "To check job status manually: squeue -u \$USER"
echo "To cancel all jobs: scancel ${job_ids[*]}"
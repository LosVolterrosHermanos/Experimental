#!/bin/bash

# Grid search script for tanea hyperparameters with clipsnr
# Parameters to vary:
# - tanea_g2: {2E-5, 4E-5, 6E-5, 8E-5}
# - clipsnr: {1.0, 2.0, 3.0, 4.0}
# - tanea_g3: 0 (fixed)
# - momentum_flavor: mk3 (fixed)
# Plus Adam baseline

# Fixed parameters
TANEA_KAPPA=0.75
WEIGHT_DECAY_TS=100
TRAIN_STEPS=500
DECAY=""
SEQ_LEN=1024
BATCH_SIZE=8

# Arrays for grid search parameters
TANEA_G2_VALUES=(2E-5 4E-5 6E-5 8E-5)
CLIPSNR_VALUES=(1.0 2.0 3.0 4.0)
TANEA_G3=0
MOMENTUM_FLAVOR="mk3"

# Counter for tracking progress
total_combinations=$(( ${#TANEA_G2_VALUES[@]} * ${#CLIPSNR_VALUES[@]} + 1 ))  # +1 for Adam baseline
current=0

echo "Starting grid search for tanea hyperparameters with clipsnr"
echo "Total combinations: $total_combinations"
echo "Parameters:"
echo "  tanea_g2: ${TANEA_G2_VALUES[*]}"
echo "  clipsnr: ${CLIPSNR_VALUES[*]}"
echo "  tanea_g3: $TANEA_G3 (fixed)"
echo "  momentum_flavor: $MOMENTUM_FLAVOR (fixed)"
echo "  Adam baseline: beta1=0.9, beta2=0.95, lr=3e-4"
echo "  Fixed: tanea_kappa=$TANEA_KAPPA, weight_decay_ts=$WEIGHT_DECAY_TS, train_steps=$TRAIN_STEPS"
echo ""

# Create results directory with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="grid_search_g2_clipsnr_results_${timestamp}"
mkdir -p "$results_dir"

# Log file for the grid search
log_file="$results_dir/grid_search_g2_clipsnr.log"

echo "Grid search started at $(date)" | tee -a "$log_file"
echo "Results will be saved to: $results_dir" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# First run Adam baseline
current=$((current + 1))
echo "=== Combination $current/$total_combinations (Adam Baseline) ===" | tee -a "$log_file"
echo "Parameters: Adam baseline with beta1=0.9, beta2=0.95, lr=3e-4" | tee -a "$log_file"
echo "Started at: $(date)" | tee -a "$log_file"

python nanogpt_adamw_baseline_mixed_bf16_rope.py \
    --train_steps="$TRAIN_STEPS" \
    --batch_size="$BATCH_SIZE" \
    --val_batch_size=1 \
    --val_steps=1 \
    --seq_len="$SEQ_LEN" \
    --lr=3E-4 \
    --beta1=0.9 \
    --beta2=0.95 \
    --weight_decay=1E-3 \
    --attention_implementation="xla" \
    --results_dir "$results_dir"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "✓ Adam baseline completed successfully" | tee -a "$log_file"
else
    echo "✗ Adam baseline failed with exit code $?" | tee -a "$log_file"
fi

echo "Finished at: $(date)" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Grid search loop for Tanea
for tanea_g2 in "${TANEA_G2_VALUES[@]}"; do
    for clipsnr in "${CLIPSNR_VALUES[@]}"; do
        current=$((current + 1))
        
        echo "=== Combination $current/$total_combinations ===" | tee -a "$log_file"
        echo "Parameters: tanea_g2=$tanea_g2, clipsnr=$clipsnr, tanea_g3=$TANEA_G3, momentum_flavor=$MOMENTUM_FLAVOR" | tee -a "$log_file"
        echo "Started at: $(date)" | tee -a "$log_file"
        
        # Run the experiment
        python nanogpt_tanea_tau_stats_mixed_bf16_rope.py \
            --train_steps="$TRAIN_STEPS" \
            --batch_size="$BATCH_SIZE" \
            --val_batch_size=1 \
            --val_steps=1 \
            --seq_len="$SEQ_LEN" \
            --tanea_g2="$tanea_g2" \
            --tanea_g3="$TANEA_G3" \
            --tanea_kappa="$TANEA_KAPPA" \
            --clipsnr="$clipsnr" \
            --weight_decay=1E-3 \
            --power_weight_decay=1.0 \
            --weight_decay_ts="$WEIGHT_DECAY_TS" \
            --momentum_flavor "$MOMENTUM_FLAVOR" \
            $DECAY \
            --attention_implementation="xla" \
            --disable_checkpoint \
            --results_dir "$results_dir"
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Completed successfully" | tee -a "$log_file"
        else
            echo "✗ Failed with exit code $?" | tee -a "$log_file"
        fi
        
        echo "Finished at: $(date)" | tee -a "$log_file"
        echo "" | tee -a "$log_file"
        
        # Optional: add a small delay between runs to avoid overwhelming the system
        sleep 2
    done
done

echo "=== Grid search completed ===" | tee -a "$log_file"
echo "Finished at: $(date)" | tee -a "$log_file"
echo "Total combinations run: $current" | tee -a "$log_file"
echo "Results directory: $results_dir" | tee -a "$log_file"
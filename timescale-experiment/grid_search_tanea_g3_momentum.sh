#!/bin/bash

# Grid search script for tanea g3 and momentum flavor parameters
# Parameters to vary:
# - tanea_g3: {1E-6, 5E-6, 1E-5, 5E-5, 1E-4}
# - momentum_flavor: {effective-clip, mk2, mk3}
# - tanea_g2: (fixed from previous sweep)
# - clipsnr: (fixed from previous sweep)
# Plus Adam baseline

# Fixed parameters (using best values from previous sweep)
TANEA_KAPPA=0.75
WEIGHT_DECAY_TS=100
TRAIN_STEPS=20
DECAY=""
SEQ_LEN=1024
BATCH_SIZE=8
VAL_BATCH_SIZE=8
VAL_STEPS=8
CLIP_NORM=100.0

# Fixed parameters from previous sweep (update these based on your best results)
TANEA_G2=1E-4
CLIPSNR=1.0

# Arrays for grid search parameters
TANEA_G3_VALUES=(8E-5 4E-5 2E-5)
MOMENTUM_FLAVOR_VALUES=("effective-clip" "mk2" "mk3")

# Momentum flavor scalers
MK2_SCALER=$(echo "scale=10; 11480/9332" | bc -l)
MK3_SCALER=$(echo "scale=10; 11480/7198" | bc -l)

# Counter for tracking progress
total_combinations=$(( ${#TANEA_G3_VALUES[@]} * ${#MOMENTUM_FLAVOR_VALUES[@]} + 2 ))  # +2 for two Adam baselines
current=0

echo "Starting grid search for tanea g3 and momentum flavor parameters"
echo "Total combinations: $total_combinations"
echo "Parameters:"
echo "  tanea_g3: ${TANEA_G3_VALUES[*]}"
echo "  momentum_flavor: ${MOMENTUM_FLAVOR_VALUES[*]}"
echo "  tanea_g2: $TANEA_G2 (fixed)"
echo "  clipsnr: $CLIPSNR (fixed)"
echo "  Adam baseline 1: beta1=0.9, beta2=0.95, lr=3e-4"
echo "  Adam baseline 2: beta1=0.0, beta2=0.95, lr=3e-4"
echo "  Fixed: tanea_kappa=$TANEA_KAPPA, weight_decay_ts=$WEIGHT_DECAY_TS, train_steps=$TRAIN_STEPS"
echo ""

# Create results directory with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="grid_search_g3_momentum_results_${timestamp}"
mkdir -p "$results_dir"

# Log file for the grid search
log_file="$results_dir/grid_search_g3_momentum.log"

echo "Grid search started at $(date)" | tee -a "$log_file"
echo "Results will be saved to: $results_dir" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# First run Adam baseline (beta1=0.9)
current=$((current + 1))
echo "=== Combination $current/$total_combinations (Adam Baseline beta1=0.9) ===" | tee -a "$log_file"
echo "Parameters: Adam baseline with beta1=0.9, beta2=0.95, lr=3e-4" | tee -a "$log_file"
echo "Started at: $(date)" | tee -a "$log_file"

python nanogpt_adamw_baseline_mixed_bf16_rope.py \
    --train_steps="$TRAIN_STEPS" \
    --batch_size="$BATCH_SIZE" \
    --val_batch_size="$VAL_BATCH_SIZE" \
    --val_steps="$VAL_STEPS" \
    --seq_len="$SEQ_LEN" \
    --lr=3E-4 \
    --beta1=0.9 \
    --beta2=0.95 \
    --grad_clip="$CLIP_NORM" \
    --weight_decay=1E-3 \
    --attention_implementation="xla" \
    --results_dir "$results_dir"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "✓ Adam baseline (beta1=0.9) completed successfully" | tee -a "$log_file"
else
    echo "✗ Adam baseline (beta1=0.9) failed with exit code $?" | tee -a "$log_file"
fi

echo "Finished at: $(date)" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Second run Adam baseline (beta1=0.0)
current=$((current + 1))
echo "=== Combination $current/$total_combinations (Adam Baseline beta1=0.0) ===" | tee -a "$log_file"
echo "Parameters: Adam baseline with beta1=0.0, beta2=0.95, lr=3e-4" | tee -a "$log_file"
echo "Started at: $(date)" | tee -a "$log_file"

python nanogpt_adamw_baseline_mixed_bf16_rope.py \
    --train_steps="$TRAIN_STEPS" \
    --batch_size="$BATCH_SIZE" \
    --val_batch_size="$VAL_BATCH_SIZE" \
    --val_steps="$VAL_STEPS" \
    --seq_len="$SEQ_LEN" \
    --lr=8E-5 \
    --beta1=0.0 \
    --beta2=0.95 \
    --grad_clip="$CLIP_NORM" \
    --weight_decay=1E-3 \
    --attention_implementation="xla" \
    --results_dir "$results_dir"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "✓ Adam baseline (beta1=0.0) completed successfully" | tee -a "$log_file"
else
    echo "✗ Adam baseline (beta1=0.0) failed with exit code $?" | tee -a "$log_file"
fi

echo "Finished at: $(date)" | tee -a "$log_file"
echo "" | tee -a "$log_file"

# Grid search loop for Tanea
for tanea_g3 in "${TANEA_G3_VALUES[@]}"; do
    for momentum_flavor in "${MOMENTUM_FLAVOR_VALUES[@]}"; do
        current=$((current + 1))
        
        # Apply momentum flavor scaling to g3
        if [ "$momentum_flavor" = "mk2" ]; then
            scaled_g3=$(python3 -c "print(f'{float('$tanea_g3') * $MK2_SCALER:.3e}')")
        elif [ "$momentum_flavor" = "mk3" ]; then
            scaled_g3=$(python3 -c "print(f'{float('$tanea_g3') * $MK3_SCALER:.3e}')")
        else
            scaled_g3=$tanea_g3
        fi
        
        echo "=== Combination $current/$total_combinations ===" | tee -a "$log_file"
        echo "Parameters: tanea_g3=$tanea_g3 (scaled: $scaled_g3), momentum_flavor=$momentum_flavor, tanea_g2=$TANEA_G2, clipsnr=$CLIPSNR" | tee -a "$log_file"
        echo "Started at: $(date)" | tee -a "$log_file"
        
        # Run the experiment
        python nanogpt_tanea_tau_stats_mixed_bf16_rope.py \
            --train_steps="$TRAIN_STEPS" \
            --batch_size="$BATCH_SIZE" \
            --val_batch_size="$VAL_BATCH_SIZE" \
            --val_steps="$VAL_STEPS" \
            --seq_len="$SEQ_LEN" \
            --tanea_g2="$TANEA_G2" \
            --tanea_g3="$scaled_g3" \
            --tanea_kappa="$TANEA_KAPPA" \
            --clipsnr="$CLIPSNR" \
            --grad_clip="$CLIP_NORM" \
            --weight_decay=1E-3 \
            --power_weight_decay=1.0 \
            --weight_decay_ts="$WEIGHT_DECAY_TS" \
            --momentum_flavor "$momentum_flavor" \
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
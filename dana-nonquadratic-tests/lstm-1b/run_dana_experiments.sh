#!/bin/bash

# Script to run LSTM model with different DANA g3_p parameter values
# This script will run the model with g3_p values: -0.5, -0.4, -0.6, -0.45, -0.55
# Note: With max_tokens=0 (full dataset), data loading may take 5-10 minutes before
#       training begins. The added debug information will show progress.

# Create a directory for logs if it doesn't exist
mkdir -p logs

# Array of g3_p values to test
g3_p_values=(-0.5 -0.4 -0.6 -0.45 -0.55)  

# Run the model for each g3_p value
for p in "${g3_p_values[@]}"; do
    echo "Running model with DANA g3_p = $p"
    logfile="logs/dana_g3_p_${p}.log"
    
    # Clear previous log file if it exists
    > "$logfile"
    
    echo "=== Starting run with g3_p = $p at $(date) ===" | tee -a "$logfile"
    
    # Run with max_tokens=0 (full dataset) but fewer training steps
    # The added debug output will show progress during data loading
    python flax-billion-word-lstm.py --train_steps=10000 --max_tokens=0 --dana_g3_p=$p --optimizer=dana | tee -a "$logfile"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed run with g3_p = $p" | tee -a "$logfile"
    else
        echo "Error running model with g3_p = $p" | tee -a "$logfile"
    fi
    
    echo "=== Finished run with g3_p = $p at $(date) ===" | tee -a "$logfile"
    echo "----------------------------------------"
done

echo "All experiments completed!" 

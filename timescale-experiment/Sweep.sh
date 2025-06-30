#! /bin/bash

while true
do
    # Generate random parameters within factors of 2 of the original values
    tanea_g2=$(echo "scale=6; $((RANDOM % 4000 + 1000)) / 20000000" | bc)
    tanea_g3=$(echo "scale=7; $((RANDOM % 4000 + 1000)) / 100000000" | bc)
    tanea_kappa=$(echo "scale=2; $((RANDOM % 40 + 60)) / 100" | bc)
    weight_decay_ts=$(echo "scale=0; $((RANDOM % 500))" | bc)
    
    # Current leader tanea_g2=2.0E-4, tanea_g3=3.3E-5 tanea_kappa=0.69, weight_decay_ts=500
    echo "Running with tanea_g2=$tanea_g2, tanea_g3=$tanea_g3, tanea_kappa=$tanea_kappa, weight_decay_ts=$weight_decay_ts"
    python nanogpt_tanea_tau_stats_mixed_bf16_rope.py --train_steps=30000 --batch_size=8 --val_batch_size=1 --val_steps=128 --seq_len=1024 --tanea_g2=$tanea_g2 --tanea_g3=$tanea_g3 --tanea_kappa=$tanea_kappa --weight_decay=1E-3 --power_weight_decay=1.0 --weight_decay_ts=$weight_decay_ts
done

#!/usr/bin/env python
"""
Example script to run the new nanogpt_tanea_fsdp.py with different configurations.
"""

import subprocess
import sys

def run_training(config_name, override_params=None):
    """Run training with specified config and optional parameter overrides."""
    cmd = [sys.executable, "nanogpt_tanea_fsdp.py", f"--config-name={config_name}"]
    
    if override_params:
        for param in override_params:
            cmd.append(param)
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    # Example 1: Local testing with FSDP disabled
    print("=== Example 1: Local testing ===")
    run_training("local")
    
    # Example 2: Multi-GPU with FSDP enabled
    print("\n=== Example 2: Multi-GPU FSDP ===")
    run_training("multi_gpu")
    
    # Example 3: Custom parameters
    print("\n=== Example 3: Custom parameters ===")
    run_training("nanogpt_tanea_fsdp", [
        "tanea.g2=2e-4",
        "tanea.g3=3e-5", 
        "model.fsdp_enabled=true",
        "training.batch_size_train=16",
        "wandb.mode=offline"
    ])
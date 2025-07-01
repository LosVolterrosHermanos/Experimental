#!/usr/bin/env python
"""
Script to visualize how loss at ~30k steps varies with Tanea hyperparameters,
with extrapolated losses at 45k, 60k, 75k steps using linear fits.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import argparse

def load_tanea_results(results_dir="results", pattern="*tanea_results*.pkl"):
    """Load Tanea training results from pickle files."""
    
    pickle_files = glob.glob(os.path.join(results_dir, pattern))
    
    if not pickle_files:
        raise ValueError(f"No Tanea results pickle files found in {results_dir} with pattern {pattern}")
    
    print(f"Found {len(pickle_files)} Tanea results files")
    
    results_data = []
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract relevant information
            config = data['config']
            metrics = data['metrics']
            num_params = data.get('num_params', 0)
            
            results_data.append({
                'config': config,
                'metrics': metrics,
                'num_params': num_params,
                'filename': os.path.basename(pkl_file)
            })
            print(f"Loaded data from {os.path.basename(pkl_file)}")
            print(f"  Parameters: g2={config['tanea_g2']}, g3={config['tanea_g3']}, kappa={config['tanea_kappa']}")
        
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    if not results_data:
        raise ValueError("No valid Tanea results data found")
    
    return results_data

def create_hyperparameter_loss_plots(results_data, output_file="hyperparameter_loss_analysis.pdf"):
    """Create plots showing loss vs hyperparameters at different steps with extrapolation."""
    
    # Extract hyperparameter data
    hyperparams = []
    target_step = 30000
    extrapolate_steps = [45000, 60000, 75000]
    
    for data in results_data:
        config = data['config']
        metrics = data['metrics']
        steps = np.array(metrics['step'])
        val_losses = np.array(metrics['val_loss'])
        
        # Find step closest to but less than 30000
        valid_steps = steps[steps < target_step]
        if len(valid_steps) == 0:
            print(f"Warning: No steps < {target_step} found for config with g2={config['tanea_g2']}")
            continue
        
        target_idx = np.where(steps == valid_steps[-1])[0][0]
        actual_step = steps[target_idx]
        loss_at_target = val_losses[target_idx]
        
        print(f"Using step {actual_step} (closest to {target_step}) for g2={config['tanea_g2']}")
        
        # Get previous 4 points for linear fit (log-log space)
        extrapolated_losses = []
        if target_idx >= 4:
            fit_steps = steps[target_idx-3:target_idx+1]
            fit_losses = val_losses[target_idx-3:target_idx+1]
            
            # Linear fit in log-log space: log(loss) = a * log(step) + b
            log_steps = np.log(fit_steps)
            log_losses = np.log(fit_losses)
            
            # Linear regression
            A = np.vstack([log_steps, np.ones(len(log_steps))]).T
            slope, intercept = np.linalg.lstsq(A, log_losses, rcond=None)[0]
            
            print(f"  Linear fit: slope={slope:.4f}, intercept={intercept:.4f}")
            
            # Extrapolate to future steps
            for step in extrapolate_steps:
                log_extrapolated = slope * np.log(step) + intercept
                extrapolated_losses.append(np.exp(log_extrapolated))
        else:
            print(f"  Warning: Not enough points for extrapolation (only {target_idx+1} points)")
            extrapolated_losses = [None, None, None]
        
        # Store hyperparameter values
        g2 = config['tanea_g2']
        kappa = config['tanea_kappa']
        g3 = config['tanea_g3']
        log_wd_delta = config['weight_decay_ts'] * config['weight_decay']
        
        hyperparams.append({
            'g2': g2,
            'kappa': kappa,
            'g3': g3,
            'log_wd_delta': log_wd_delta,
            'loss_at_target': loss_at_target,
            'actual_step': actual_step,
            'extrapolated_losses': extrapolated_losses
        })
    
    if not hyperparams:
        print("No valid data found for hyperparameter analysis")
        return
    
    print(f"\nAnalyzing {len(hyperparams)} configurations")
    
    # Create 2x2 subplot for the 4 hyperparameters
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    hyperparam_names = ['g2', 'kappa', 'g3', 'log_wd_delta']
    hyperparam_labels = ['g₂', 'κ', 'g₃', 'log(wd·δ)']
    
    # Colors for different steps using plasma colormap (dark to light)
    colors = plt.cm.plasma([0.0, 0.33, 0.67, 1.0])
    
    for i, (param_name, param_label) in enumerate(zip(hyperparam_names, hyperparam_labels)):
        ax = axes[i]
        
        # Extract parameter values and losses
        param_values = [h[param_name] for h in hyperparams]
        
        # Plot loss at target step (~30k)
        target_losses = [h['loss_at_target'] for h in hyperparams]
        actual_steps = [h['actual_step'] for h in hyperparams]
        
        ax.scatter(param_values, target_losses, c=[colors[0]], s=80, 
                  label=f'~{target_step//1000}k steps', alpha=0.8, 
                  edgecolors='black', linewidth=0.5, zorder=3)
        
        # Plot extrapolated losses
        for j, step in enumerate(extrapolate_steps):
            extrap_losses = [h['extrapolated_losses'][j] for h in hyperparams 
                           if h['extrapolated_losses'][j] is not None]
            extrap_params = [h[param_name] for h in hyperparams 
                           if h['extrapolated_losses'][j] is not None]
            
            if extrap_losses:
                ax.scatter(extrap_params, extrap_losses, c=[colors[j+1]], s=80, 
                          label=f'{step//1000}k steps (extrap)', alpha=0.8, 
                          edgecolors='black', linewidth=0.5, zorder=3)
        
        ax.set_xlabel(param_label, fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(f'Loss vs {param_label}', fontsize=13)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.legend(fontsize=10)
        
        # Use log scale for parameters that span orders of magnitude
        if param_name in ['g2', 'g3']:
            ax.set_xscale('log')
        
        # Connect points with lines to show trends
        if len(param_values) > 1:
            # Sort by parameter value for line connections
            sorted_indices = np.argsort(param_values)
            sorted_params = [param_values[idx] for idx in sorted_indices]
            sorted_losses = [target_losses[idx] for idx in sorted_indices]
            
            ax.plot(sorted_params, sorted_losses, 'k--', alpha=0.3, zorder=2, linewidth=1)
            
            # Also connect extrapolated points
            for j, step in enumerate(extrapolate_steps):
                extrap_losses = []
                extrap_params = []
                for idx in sorted_indices:
                    if hyperparams[idx]['extrapolated_losses'][j] is not None:
                        extrap_losses.append(hyperparams[idx]['extrapolated_losses'][j])
                        extrap_params.append(hyperparams[idx][param_name])
                
                if len(extrap_losses) > 1:
                    ax.plot(extrap_params, extrap_losses, '--', color=colors[j+1], 
                           alpha=0.3, zorder=2, linewidth=1)
    
    plt.suptitle('NanoGPT Tanea: Loss vs Hyperparameters\n(~30k actual + 45k/60k/75k extrapolated)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nHyperparameter loss plots saved as {output_file}")
    plt.show()

def print_hyperparameter_summary(hyperparams):
    """Print summary of hyperparameter analysis."""
    
    print("\n" + "="*80)
    print("Hyperparameter Loss Analysis Summary")
    print("="*80)
    
    target_step = 30000
    extrapolate_steps = [45000, 60000, 75000]
    
    print(f"\nAnalyzing {len(hyperparams)} configurations:")
    print(f"Target step: ~{target_step}")
    print(f"Extrapolated steps: {extrapolate_steps}")
    
    for i, h in enumerate(hyperparams):
        print(f"\nConfig {i+1}:")
        print(f"  g2={h['g2']:.2e}, κ={h['kappa']:.3f}, g3={h['g3']:.2e}, log(wd·δ)={h['log_wd_delta']:.4f}")
        print(f"  Loss at step {h['actual_step']}: {h['loss_at_target']:.6f}")
        
        if h['extrapolated_losses'][0] is not None:
            print(f"  Extrapolated losses:")
            for j, step in enumerate(extrapolate_steps):
                if h['extrapolated_losses'][j] is not None:
                    print(f"    Step {step}: {h['extrapolated_losses'][j]:.6f}")

def main():
    """Main function to load data and create hyperparameter visualizations."""
    parser = argparse.ArgumentParser(description="Plot hyperparameter vs loss analysis for NanoGPT Tanea")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing results pickle files")
    parser.add_argument("--pattern", type=str, default="*tanea_results*.pkl",
                       help="Pattern to match result files")
    parser.add_argument("--output", type=str, default="hyperparameter_loss_analysis.pdf",
                       help="Output file name")
    
    args = parser.parse_args()
    
    try:
        # Load results data
        if not os.path.exists(args.results_dir):
            print(f"Results directory '{args.results_dir}' not found!")
            return
        
        results_data = load_tanea_results(args.results_dir, args.pattern)
        print(f"\nLoaded results for {len(results_data)} Tanea configurations")
        
        # Create hyperparameter analysis
        create_hyperparameter_loss_plots(results_data, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Simple visualization script for tanea grid search results.
Plots loss curves for all combinations including Adam baseline.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

def load_results(results_dir):
    """Load all results from the results directory."""
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(results_dir, "*.pkl"))
    
    if not pickle_files:
        raise ValueError(f"No pickle files found in {results_dir}")
    
    print(f"Found {len(pickle_files)} result files")
    
    results_data = []
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract relevant information
            config = data['config']
            metrics = data['metrics']
            filename = os.path.basename(pkl_file)
            
            # Determine if this is Adam baseline or Tanea
            if 'adamw_baseline' in filename:
                optimizer_type = 'Adam'
                g2_value = None
                g3_value = None
                label = f"Adam (lr={config['lr']}, β1={config['beta1']}, β2={config['beta2']})"
            else:
                optimizer_type = 'Tanea'
                g2_value = config['tanea_g2']
                g3_value = config['tanea_g3']
                flavor = config['momentum_flavor']
                label = f"Tanea (g2={g2_value}, g3={g3_value}, {flavor})"
            
            results_data.append({
                'config': config,
                'metrics': metrics,
                'filename': filename,
                'optimizer_type': optimizer_type,
                'g2_value': g2_value,
                'g3_value': g3_value,
                'label': label,
                'steps': np.array(metrics['step']),
                'train_losses': np.array(metrics['train_loss']),
                'val_losses': np.array(metrics['val_loss'])
            })
            
            print(f"Loaded: {label}")
        
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    if not results_data:
        raise ValueError("No valid results data found")
    
    print(f"\nSuccessfully loaded {len(results_data)} results")
    return results_data

def plot_loss_curves(results_data, output_dir):
    """Plot all loss curves."""
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for different optimizers
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_data)))
    
    for i, result in enumerate(results_data):
        steps = result['steps']
        train_losses = result['train_losses']
        val_losses = result['val_losses']
        label = result['label']
        
        # Plot training loss
        plt.plot(steps, train_losses, color=colors[i], linewidth=2, 
                label=f"{label} (train)", alpha=0.8)
        
        # Plot validation loss if available and not all NaN
        if not np.isnan(val_losses).all():
            plt.plot(steps, val_losses, color=colors[i], linewidth=2, 
                    linestyle='--', label=f"{label} (val)", alpha=0.6)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Use log scale for y-axis if losses vary significantly
    all_losses = []
    for result in results_data:
        all_losses.extend(result['train_losses'])
        if not np.isnan(result['val_losses']).all():
            all_losses.extend(result['val_losses'])
    
    if max(all_losses) / min(all_losses) > 10:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_summary(results_data):
    """Print summary statistics."""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Separate Adam and Tanea results
    adam_results = [r for r in results_data if r['optimizer_type'] == 'Adam']
    tanea_results = [r for r in results_data if r['optimizer_type'] == 'Tanea']
    
    # Adam results
    if adam_results:
        print("\nAdam Baseline:")
        for result in adam_results:
            final_train = result['train_losses'][-1]
            final_val = result['val_losses'][-1] if not np.isnan(result['val_losses']).all() else float('nan')
            print(f"  Final train loss: {final_train:.6f}")
            print(f"  Final val loss: {final_val:.6f}")
    
    # Tanea results
    if tanea_results:
        print("\nTanea Results (g3=0, mk3):")
        print(f"{'g2':<10} {'Train Loss':<12} {'Val Loss':<12}")
        print("-" * 40)
        
        for result in sorted(tanea_results, key=lambda x: x['g2_value']):
            final_train = result['train_losses'][-1]
            final_val = result['val_losses'][-1] if not np.isnan(result['val_losses']).all() else float('nan')
            print(f"{result['g2_value']:<10.0e} {final_train:<12.6f} {final_val:<12.6f}")
        
        # Find best Tanea configuration
        best_tanea = min(tanea_results, key=lambda x: x['train_losses'][-1])
        print(f"\nBest Tanea configuration:")
        print(f"  g2={best_tanea['g2_value']}")
        print(f"  Final train loss: {best_tanea['train_losses'][-1]:.6f}")
        print(f"  Final val loss: {best_tanea['val_losses'][-1]:.6f}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple visualization of grid search results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing results")
    parser.add_argument("--output_dir", type=str, default="simple_visualizations",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load results
        if not os.path.exists(args.results_dir):
            print(f"Results directory '{args.results_dir}' not found!")
            return
        
        results_data = load_results(args.results_dir)
        
        # Create visualization
        print("\nCreating loss curves...")
        plot_loss_curves(results_data, args.output_dir)
        
        # Print summary
        print_summary(results_data)
        
        print(f"\nVisualization saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
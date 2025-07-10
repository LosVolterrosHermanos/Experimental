#!/usr/bin/env python3

import re
import glob
import os
import time
import math
from collections import defaultdict, deque
import sys
import signal
from rich.console import Console
from rich.live import Live
from rich.text import Text

class TrainingMonitor:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.data = defaultdict(lambda: {'iterations': deque(), 'losses': deque()})
        self.file_positions = {}
        self.optimizer_labels = {}
        self.running = True
        self.console = Console()
        
        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Initialize file tracking
        self.initialize_files()
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.running = False
        print("\n\nStopping monitor...")
        
    def initialize_files(self):
        """Initialize file tracking and get optimizer labels"""
        err_files = glob.glob(os.path.join(self.results_dir, '*.err'))
        
        # Famous horse racing names mapped to optimizer types
        horse_names = {
            'adamw': ['Secretariat', 'Seabiscuit', 'Man o War', 'Citation', 'Kelso'],
            'tanea_g2': ['Spectacular Bid', 'Affirmed', 'Alydar', 'Seattle Slew', 'Forego'],
            'tanea_g3': ['War Admiral', 'Count Fleet', 'Whirlaway', 'Assault', 'Omaha'],
            'tanea_kappa': ['American Pharoah', 'Justify', 'California Chrome', 'Smarty Jones', 'Funny Cide']
        }
        
        # Counter for assigning unique horse names
        horse_counters = {key: 0 for key in horse_names.keys()}
        
        for filepath in err_files:
            self.file_positions[filepath] = 0
            
            # Extract optimizer type from filename and file content
            filename = os.path.basename(filepath)
            
            if filename.startswith('adamw_'):
                # AdamW WSD baseline - extract learning rate multiplier
                horse_type = 'adamw'
                horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                horse_counters[horse_type] += 1
                
                # Extract learning rate multiplier from filename
                if '2x_lr' in filename:
                    lr_info = 'lr=3.2e-4'
                elif '4x_lr' in filename:
                    lr_info = 'lr=6.4e-4'
                else:
                    lr_info = 'lr=2.4e-4'  # baseline
                
                self.optimizer_labels[filepath] = f'{horse_name} (AdamW-WSD-{lr_info})'
                
            elif filename.startswith('adam_'):
                # Extract beta1 parameter from filename
                beta1_match = re.search(r'adam_b1_([0-9.]+)', filename)
                if beta1_match:
                    beta1 = beta1_match.group(1)
                    horse_type = 'adamw'
                    horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                    horse_counters[horse_type] += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Adam-b1_{beta1})'
                else:
                    horse_type = 'adamw'
                    horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                    horse_counters[horse_type] += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Adam)'
                    
            elif filename.startswith('tanea_'):
                # Handle different tanea variants - g2, g3, and kappa combinations
                g2_g3_kappa_match = re.search(r'tanea_g2_([0-9E-]+)_g3_([0-9E-]+)_kappa_([0-9.]+)_mk3_wsd_', filename)
                g2_g3_match = re.search(r'tanea_g2_([0-9E-]+)_g3_([0-9E-]+)_mk3_wsd_', filename)
                g2_clipsnr_match = re.search(r'tanea_g2_([0-9E-]+)_clipsnr_([0-9E-]+)', filename)
                g3_match = re.search(r'tanea_g3_([0-9E-]+)', filename)
                kappa_match = re.search(r'tanea_kappa_([0-9.]+)', filename)
                g3_kappa_match = re.search(r'tanea_g3_([0-9E-]+)_kappa_([0-9.]+)', filename)
                
                if g2_g3_kappa_match:
                    # g2, g3, and kappa format
                    g2_val = g2_g3_kappa_match.group(1)
                    g3_val = g2_g3_kappa_match.group(2)
                    kappa_val = g2_g3_kappa_match.group(3)
                    horse_type = 'tanea_g3'
                    horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                    horse_counters[horse_type] += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-g2_{g2_val}-g3_{g3_val}-kappa_{kappa_val})'
                elif g2_g3_match:
                    # g2 and g3 format (no kappa)
                    g2_val = g2_g3_match.group(1)
                    g3_val = g2_g3_match.group(2)
                    horse_type = 'tanea_g3'
                    horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                    horse_counters[horse_type] += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-g2_{g2_val}-g3_{g3_val})'
                elif g2_clipsnr_match:
                    # g2 clipsnr format
                    g2_val = g2_clipsnr_match.group(1)
                    clipsnr_val = g2_clipsnr_match.group(2)
                    horse_type = 'tanea_g2'
                    horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                    horse_counters[horse_type] += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-g2_{g2_val}-clipsnr_{clipsnr_val})'
                elif g3_kappa_match:
                    # g3 with kappa format
                    g3_val = g3_kappa_match.group(1)
                    kappa_val = g3_kappa_match.group(2)
                    horse_type = 'tanea_g3'
                    horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                    horse_counters[horse_type] += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-g3_{g3_val}-kappa_{kappa_val})'
                elif kappa_match:
                    # kappa only format
                    kappa_val = kappa_match.group(1)
                    horse_type = 'tanea_kappa'
                    horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                    horse_counters[horse_type] += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-kappa_{kappa_val})'
                elif g3_match:
                    # g3 momentum format
                    g3_val = g3_match.group(1)
                    
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            horse_type = 'tanea_g3'
                            horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                            horse_counters[horse_type] += 1
                            
                            if 'momentum_flavor=effective-clip' in content:
                                self.optimizer_labels[filepath] = f'{horse_name} (Tanea-eff-clip-g3_{g3_val})'
                            elif 'momentum_flavor=mk3' in content:
                                self.optimizer_labels[filepath] = f'{horse_name} (Tanea-mk3-g3_{g3_val})'
                            else:
                                self.optimizer_labels[filepath] = f'{horse_name} (Tanea-unknown-g3_{g3_val})'
                    except:
                        horse_type = 'tanea_g3'
                        horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                        horse_counters[horse_type] += 1
                        self.optimizer_labels[filepath] = f'{horse_name} (Unknown-g3_{g3_val})'
                else:
                    # Fallback for unknown tanea format
                    horse_type = 'tanea_g3'
                    horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                    horse_counters[horse_type] += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-unknown)'
            else:
                # Unknown optimizer type
                horse_type = 'adamw'
                horse_name = horse_names[horse_type][horse_counters[horse_type] % len(horse_names[horse_type])]
                horse_counters[horse_type] += 1
                self.optimizer_labels[filepath] = f'{horse_name} (Unknown)'
    
    def parse_training_line(self, line):
        """Parse a training progress line to extract iteration and loss"""
        # Look for training progress lines
        iter_match = re.search(r'Training:\s*\d+%\|[^|]*\|\s*(\d+)/\d+', line)
        if not iter_match:
            return None, None
            
        iteration = int(iter_match.group(1))
        
        # Find the loss value
        loss_match = re.search(r'loss=([0-9.]+)', line)
        if not loss_match:
            return None, None
            
        loss = float(loss_match.group(1))
        
        return iteration, loss
    
    def update_data(self):
        """Update data by reading new lines from all files"""
        new_data_found = False
        
        for filepath in self.file_positions:
            if not os.path.exists(filepath):
                continue
                
            try:
                with open(filepath, 'r') as f:
                    # Seek to last position
                    f.seek(self.file_positions[filepath])
                    
                    # Read new lines
                    new_lines = f.readlines()
                    
                    # Update position
                    self.file_positions[filepath] = f.tell()
                    
                    # Process new lines
                    for line in new_lines:
                        iteration, loss = self.parse_training_line(line)
                        if iteration is not None and loss is not None:
                            optimizer = self.optimizer_labels[filepath]
                            self.data[optimizer]['iterations'].append(iteration)
                            self.data[optimizer]['losses'].append(loss)
                            new_data_found = True
                            
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                
        return new_data_found
    
    def abbreviate_optimizer_name(self, name):
        """Abbreviate optimizer names for display"""
        # Extract horse name (first word) and optimizer info (in parentheses)
        if '(' in name and ')' in name:
            horse_name = name.split('(')[0].strip()
            optimizer_info = name.split('(')[1].split(')')[0]
            
            # Use longer horse name abbreviation (first 8 characters)
            horse_abbrev = horse_name[:8]
            
            # Abbreviate optimizer info
            if optimizer_info.startswith('AdamW-WSD'):
                # Extract learning rate from optimizer info
                lr_match = re.search(r'lr=([0-9.e-]+)', optimizer_info)
                if lr_match:
                    lr_val = lr_match.group(1)
                    opt_abbrev = f"AdW-{lr_val}"
                else:
                    opt_abbrev = "AdW"
            elif optimizer_info.startswith('Adam-b1_'):
                beta1 = optimizer_info.split('_')[1]
                opt_abbrev = f"A-b1_{beta1}"
            elif 'Tanea-g2_' in optimizer_info and 'g3_' in optimizer_info and 'kappa_' in optimizer_info:
                # Extract g2, g3, and kappa values from new format
                g2_match = re.search(r'g2_([0-9E.-]+)', optimizer_info)
                g3_match = re.search(r'g3_([0-9E.-]+)', optimizer_info)
                kappa_match = re.search(r'kappa_([0-9.]+)', optimizer_info)
                g2_val = g2_match.group(1) if g2_match else 'unknown'
                g3_val = g3_match.group(1) if g3_match else 'unknown'
                kappa_val = kappa_match.group(1) if kappa_match else 'unknown'
                opt_abbrev = f"T-g2{g2_val}-g3{g3_val}-k{kappa_val}"
            elif 'Tanea-g2_' in optimizer_info and 'g3_' in optimizer_info:
                # Extract g2 and g3 values from new format (no kappa)
                g2_match = re.search(r'g2_([0-9E.-]+)', optimizer_info)
                g3_match = re.search(r'g3_([0-9E.-]+)', optimizer_info)
                g2_val = g2_match.group(1) if g2_match else 'unknown'
                g3_val = g3_match.group(1) if g3_match else 'unknown'
                opt_abbrev = f"T-g2{g2_val}-g3{g3_val}"
            elif 'Tanea-g2_' in optimizer_info and 'clipsnr_' in optimizer_info:
                # Extract g2 and clipsnr values
                parts = optimizer_info.split('-')
                g2_val = parts[1].split('_')[1] if len(parts) > 1 else 'unknown'
                clipsnr_val = parts[3].split('_')[1] if len(parts) > 3 else 'unknown'
                opt_abbrev = f"TG2-{g2_val}-cs{clipsnr_val}"
            elif 'Tanea-g3_' in optimizer_info and 'kappa_' in optimizer_info:
                # Extract g3 and kappa values
                g3_match = re.search(r'g3_([0-9E.-]+)', optimizer_info)
                kappa_match = re.search(r'kappa_([0-9.]+)', optimizer_info)
                g3_val = g3_match.group(1) if g3_match else 'unknown'
                kappa_val = kappa_match.group(1) if kappa_match else 'unknown'
                opt_abbrev = f"TG3-{g3_val}-k{kappa_val}"
            elif 'Tanea-kappa_' in optimizer_info:
                kappa_val = optimizer_info.split('_')[1]
                opt_abbrev = f"Tk-{kappa_val}"
            elif 'Tanea-eff-clip-g3_' in optimizer_info:
                g3_val = optimizer_info.split('_')[-1]
                opt_abbrev = f"TEF-{g3_val}"
            elif 'Tanea-mk3-g3_' in optimizer_info:
                g3_val = optimizer_info.split('_')[-1]
                opt_abbrev = f"TMK3-{g3_val}"
            else:
                opt_abbrev = optimizer_info[:8]  # Fallback truncation
                
            return f"{horse_abbrev}-{opt_abbrev}"
        else:
            return name[:20]  # Fallback truncation
    
    def get_color_for_optimizer(self, optimizer):
        """Get rich color name for optimizer type"""
        # Extract optimizer info from parentheses
        if '(' in optimizer and ')' in optimizer:
            optimizer_info = optimizer.split('(')[1].split(')')[0]
            
            if 'AdamW' in optimizer_info or 'Adam' in optimizer_info:
                return 'blue'
            elif 'Tanea-g2_' in optimizer_info:
                return 'magenta'
            elif 'Tanea-eff-clip' in optimizer_info:
                return 'green'
            elif 'Tanea-mk3' in optimizer_info:
                return 'red'
            elif 'Tanea-kappa' in optimizer_info:
                return 'yellow'
            elif 'Tanea-g3' in optimizer_info:
                return 'cyan'
            else:
                return 'white'
        else:
            return 'white'
    
    def calculate_power_weighted_moving_average(self, iterations, losses, current_iter):
        """Calculate power-weighted moving average using sum_{i=1}^t f(i) * (2 * i/(t*(t+1)))"""
        if not iterations or not losses:
            return None
        
        t = len(iterations)
        if t == 0:
            return None
        
        # Calculate weighted sum
        weighted_sum = 0
        for i in range(t):
            # Weight is 2 * (i+1) / (t * (t+1))
            weight = 2 * (i + 1) / (t * (t + 1))
            weighted_sum += losses[i] * weight
        
        return weighted_sum
    
    def calculate_moving_averages_at_intervals(self, iterations, losses, current_iter):
        """Calculate moving averages at current time, iter/1.1, iter/1.1^2, etc."""
        if not iterations or not losses or len(iterations) == 0:
            return {}
        
        # Convert to lists for easier manipulation
        iter_list = list(iterations)
        loss_list = list(losses)
        
        # Calculate moving averages at different time intervals
        intervals = {}
        for power in range(4):  # 0 to 3 (iter, iter/1.1, iter/1.1^2, iter/1.1^3)
            target_iter = current_iter / (1.1 ** power)
            
            # Find data points up to this target iteration
            valid_indices = [i for i, it in enumerate(iter_list) if it <= target_iter]
            
            if valid_indices:
                # Get the relevant data
                relevant_iters = [iter_list[i] for i in valid_indices]
                relevant_losses = [loss_list[i] for i in valid_indices]
                
                # Calculate power-weighted moving average
                moving_avg = self.calculate_power_weighted_moving_average(relevant_iters, relevant_losses, target_iter)
                
                if moving_avg is not None:
                    intervals[f"iter/{1.1**power:.3f}"] = {
                        'target_iter': target_iter,
                        'moving_avg': moving_avg,
                        'data_points': len(relevant_losses)
                    }
        
        return intervals
    
    def create_scatter_plot(self):
        """Create ASCII scatter plot of iteration vs power-weighted moving average"""
        if not self.data:
            return Text("No data available yet")
        
        # Create rich Text object for the plot
        plot_text = Text()
        
        # Collect data for each optimizer
        plot_data = []
        letter_map = {}
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letter_idx = 0
        
        for optimizer in self.data:
            if len(self.data[optimizer]['iterations']) > 0:
                iterations = list(self.data[optimizer]['iterations'])
                losses = list(self.data[optimizer]['losses'])
                
                # Calculate moving averages at different time intervals
                if len(losses) > 0:
                    latest_iter = iterations[-1] if iterations else 0
                    
                    # Calculate moving averages at different intervals
                    moving_averages = self.calculate_moving_averages_at_intervals(iterations, losses, latest_iter)
                    
                    # Use the current time moving average as the main value
                    moving_avg = moving_averages.get('iter/1.000', {}).get('moving_avg', 
                                                   self.calculate_power_weighted_moving_average(iterations, losses, latest_iter))
                    
                    if moving_avg is None:
                        moving_avg = sum(losses) / len(losses)  # Fallback to simple average
                    
                    # Assign letter to optimizer
                    if optimizer not in letter_map:
                        letter_map[optimizer] = letters[letter_idx % len(letters)]
                        letter_idx += 1
                    
                    plot_data.append({
                        'optimizer': optimizer,
                        'iteration': latest_iter,
                        'moving_avg': moving_avg,
                        'letter': letter_map[optimizer],
                        'color': self.get_color_for_optimizer(optimizer),
                        'data_points': len(losses),
                        'moving_averages': moving_averages,
                        'iterations': iterations,
                        'losses': losses
                    })
        
        if not plot_data:
            return "No data available yet"
        
        # Create scatter plot
        width = 120
        height = 20
        
        # Find ranges - set bounds from iter/(1.1**3) (min over all algs) to iter (max over all algs)
        all_current_iters = [data['iteration'] for data in plot_data]
        max_iter = max(all_current_iters)
        min_iter = min(data['iteration'] / (1.1**3) for data in plot_data)
        
        # Find moving average range across all data
        all_avgs = []
        for data in plot_data:
            all_avgs.append(data['moving_avg'])
            # Add all moving average intervals
            if 'moving_averages' in data and data['moving_averages']:
                for interval_name, interval_data in data['moving_averages'].items():
                    all_avgs.append(interval_data['moving_avg'])
        
        min_avg = min(all_avgs)
        max_avg = max(all_avgs)
        
        # Create plot grid - store both character and color info
        plot_grid = [[{'char': ' ', 'color': 'white'} for _ in range(width)] for _ in range(height)]
        
        # Sort plot_data by iteration in descending order (rightmost to leftmost)
        sorted_plot_data = sorted(plot_data, key=lambda x: x['iteration'], reverse=True)
        
        # Plot each optimizer in order from rightmost to leftmost
        for data in sorted_plot_data:
            # Generate continuous line of lowercase letters from current iteration to full screen bounds
            current_iter = data['iteration']
            
            # Create many interpolation points to ensure continuous line across full screen
            num_points = width * 2  # Use more points for smoother lines
            
            for i in range(num_points):
                # Linear interpolation from current_iter to min_iter (full screen bounds)
                t = i / (num_points - 1) if num_points > 1 else 0
                target_iter = current_iter - t * (current_iter - min_iter)
                
                # Find the moving average at this target iteration
                # Use data up to target_iter to calculate moving average
                valid_indices = [idx for idx, it in enumerate(data['iterations']) if it <= target_iter]
                
                if valid_indices:
                    relevant_iters = [data['iterations'][idx] for idx in valid_indices]
                    relevant_losses = [data['losses'][idx] for idx in valid_indices]
                    
                    # Calculate power-weighted moving average
                    moving_avg = self.calculate_power_weighted_moving_average(relevant_iters, relevant_losses, target_iter)
                    
                    if moving_avg is not None:
                        # Convert to plot coordinates
                        if max_iter > min_iter:
                            x = int((target_iter - min_iter) / (max_iter - min_iter) * (width - 1))
                        else:
                            x = 0
                            
                        if max_avg > min_avg:
                            y = int((max_avg - moving_avg) / (max_avg - min_avg) * (height - 1))
                        else:
                            y = height // 2
                            
                        x = max(0, min(width - 1, x))
                        y = max(0, min(height - 1, y))
                        
                        # Use uppercase for current point (i == 0), lowercase for historical points
                        char = data['letter'] if i == 0 else data['letter'].lower()
                        
                        # Overwrite existing letters (plot in order ensures rightmost algorithms overwrite leftmost)
                        plot_grid[y][x] = {'char': char, 'color': data['color']}
        
        # Create plot using rich Text
        plot_text.append("Power-Weighted Moving Average vs Iteration (continuous line visualization)\n")
        plot_text.append(f"Max Avg: {max_avg:.3f}\n")
        
        for row in plot_grid:
            plot_text.append("|")
            for cell in row:
                if cell['char'] != ' ':
                    plot_text.append(cell['char'], style=cell['color'])
                else:
                    plot_text.append(cell['char'])
            plot_text.append("|\n")
        
        plot_text.append(f"Min Avg: {min_avg:.3f}\n")
        plot_text.append(f"Iterations: {min_iter:.0f} to {max_iter}\n")
        
        # Add legend
        plot_text.append("\n")
        plot_text.append("Legend (UPPERCASE=current point, lowercase=continuous line to screen bounds):\n")
        for data in sorted(plot_data, key=lambda x: x['letter']):
            abbrev_name = self.abbreviate_optimizer_name(data['optimizer'])
            plot_text.append(f"  ")
            plot_text.append(data['letter'], style=data['color'])
            plot_text.append(f" = {abbrev_name} (iter:{data['iteration']}, avg:{data['moving_avg']:.3f}, {data['data_points']} pts)\n")
        
        return plot_text
    
    def create_status_display(self):
        """Create status display as rich Text"""
        status_text = Text()
        
        status_text.append("=" * 80 + "\n")
        status_text.append(f"TRAINING MONITOR - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        status_text.append("=" * 80 + "\n")
        
        # Print kappa values for g3 variants at the top
        status_text.append("KAPPA VALUES FOR G3 VARIANTS:\n")
        kappa_found = False
        for optimizer in sorted(self.data.keys()):
            if '(' in optimizer and ')' in optimizer:
                optimizer_info = optimizer.split('(')[1].split(')')[0]
                if 'kappa_' in optimizer_info:
                    kappa_match = re.search(r'kappa_([0-9.]+)', optimizer_info)
                    if kappa_match:
                        kappa_val = kappa_match.group(1)
                        horse_name = optimizer.split('(')[0].strip()
                        status_text.append(f"  {horse_name}: κ = {kappa_val}\n")
                        kappa_found = True
        
        if not kappa_found:
            status_text.append("  (No g3 variants with kappa found)\n")
        
        status_text.append("=" * 80 + "\n")
        
        # Print stats for each optimizer
        for optimizer in sorted(self.data.keys()):
            if len(self.data[optimizer]['iterations']) > 0:
                latest_iter = self.data[optimizer]['iterations'][-1]
                latest_loss = self.data[optimizer]['losses'][-1]
                count = len(self.data[optimizer]['iterations'])
                
                status_text.append(f"{optimizer:30s}: {count:3d} points, latest: iter={latest_iter:6d}, loss={latest_loss:.4f}\n")
        
        status_text.append("=" * 80 + "\n")
        status_text.append(self.create_scatter_plot())
        status_text.append("=" * 80 + "\n")
        status_text.append("Press Ctrl+C to stop monitoring\n")
        
        return status_text
        
    def start_monitoring(self, update_interval=5):
        """Start the monitoring loop using rich Live display"""
        print(f"Starting monitoring of {self.results_dir}")
        print(f"Found {len(self.file_positions)} .err files")
        print("Update interval: {} seconds".format(update_interval))
        
        # Initial load
        self.update_data()
        
        # Use rich Live display to reduce flickering
        with Live(self.create_status_display(), console=self.console, refresh_per_second=1) as live:
            while self.running:
                time.sleep(update_interval)
                
                if not self.running:
                    break
                    
                new_data = self.update_data()
                
                if new_data:
                    # Update the live display
                    live.update(self.create_status_display())
        
        print("\nMonitoring stopped.")

def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Try nanogpt_speedrun_results first, then fall back to grid_search
        potential_dirs = [
            './nanogpt_speedrun_results_20250709_181210',
            '../timescale-experiment/grid_search_g3_momentum_results_20250709_063908'
        ]
        
        results_dir = None
        for dir_path in potential_dirs:
            if os.path.exists(dir_path):
                results_dir = dir_path
                break
        
        if results_dir is None:
            print("No default results directory found!")
            print("Available directories:")
            for dir_path in potential_dirs:
                print(f"  - {dir_path} (exists: {os.path.exists(dir_path)})")
            sys.exit(1)
    
    # If results_dir doesn't start with '/', '../', or './', assume it's in current directory (ccanada)
    if not (results_dir.startswith('/') or results_dir.startswith('../') or results_dir.startswith('./')):
        results_dir = f'./{results_dir}'
    
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist!")
        sys.exit(1)
    
    monitor = TrainingMonitor(results_dir)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
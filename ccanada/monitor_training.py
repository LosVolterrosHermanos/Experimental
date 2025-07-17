#!/usr/bin/env python3

import re
import glob
import os
import time
import math
from collections import defaultdict
import sys
import signal
import argparse
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.text import Text

class TrainingMonitor:
    def __init__(self, results_dir, val_history_limit=10):
        self.results_dir = results_dir
        self.val_history_limit = val_history_limit
        self.data = defaultdict(lambda: {'iterations': None, 'losses': None, 'val_loss': None, 'current_idx': 0, 'max_steps': 96000, 'val_loss_history': []})
        self.file_positions = {}
        self.optimizer_labels = {}
        self.file_info = {}  # Store filename, jobID info
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
        
        # Famous horse racing names - assigned in order to avoid duplicates
        horse_names = [
            'Secretariat', 'Seabiscuit', 'Man o War', 'Citation', 'Kelso',
            'Spectacular Bid', 'Affirmed', 'Alydar', 'Seattle Slew', 'Forego',
            'War Admiral', 'Count Fleet', 'Whirlaway', 'Assault', 'Omaha',
            'American Pharoah', 'Justify', 'California Chrome', 'Smarty Jones', 'Funny Cide',
            'Gallant Fox', 'Triple Crown', 'Sir Barton', 'Gallant Man', 'Native Dancer',
            'Dr. Fager', 'Buckpasser', 'Gun Bow', 'Carry Back', 'Bold Ruler'
        ]
        
        # Counter for assigning unique horse names in order
        horse_counter = 0
        
        for filepath in err_files:
            # Get current file position (end of file) and extract current iteration
            try:
                with open(filepath, 'r') as f:
                    f.seek(0, 2)  # Seek to end of file
                    self.file_positions[filepath] = f.tell()
            except Exception:
                self.file_positions[filepath] = 0
            
            # Extract optimizer type from filename and file content
            filename = os.path.basename(filepath)
            
            # Extract jobID from filename (last number before .err)
            jobid_match = re.search(r'_(\d+)\.err$', filename)
            jobid = int(jobid_match.group(1)) if jobid_match else 0
            
            # Store filename info (strip jobID for display)
            filename_stripped = re.sub(r'_\d+\.err$', '', filename)
            self.file_info[filepath] = {
                'filename': filename,
                'filename_stripped': filename_stripped,
                'jobid': jobid
            }
            
            if filename.startswith('adamw_'):
                # AdamW WSD baseline - extract learning rate multiplier
                horse_name = horse_names[horse_counter % len(horse_names)]
                horse_counter += 1
                
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
                    horse_name = horse_names[horse_counter % len(horse_names)]
                    horse_counter += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Adam-b1_{beta1})'
                else:
                    horse_name = horse_names[horse_counter % len(horse_names)]
                    horse_counter += 1
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
                    horse_name = horse_names[horse_counter % len(horse_names)]
                    horse_counter += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-g2_{g2_val}-g3_{g3_val}-kappa_{kappa_val})'
                elif g2_g3_match:
                    # g2 and g3 format (no kappa)
                    g2_val = g2_g3_match.group(1)
                    g3_val = g2_g3_match.group(2)
                    horse_name = horse_names[horse_counter % len(horse_names)]
                    horse_counter += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-g2_{g2_val}-g3_{g3_val})'
                elif g2_clipsnr_match:
                    # g2 clipsnr format
                    g2_val = g2_clipsnr_match.group(1)
                    clipsnr_val = g2_clipsnr_match.group(2)
                    horse_name = horse_names[horse_counter % len(horse_names)]
                    horse_counter += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-g2_{g2_val}-clipsnr_{clipsnr_val})'
                elif g3_kappa_match:
                    # g3 with kappa format
                    g3_val = g3_kappa_match.group(1)
                    kappa_val = g3_kappa_match.group(2)
                    horse_name = horse_names[horse_counter % len(horse_names)]
                    horse_counter += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-g3_{g3_val}-kappa_{kappa_val})'
                elif kappa_match:
                    # kappa only format
                    kappa_val = kappa_match.group(1)
                    horse_name = horse_names[horse_counter % len(horse_names)]
                    horse_counter += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-kappa_{kappa_val})'
                elif g3_match:
                    # g3 momentum format
                    g3_val = g3_match.group(1)
                    
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            horse_name = horse_names[horse_counter % len(horse_names)]
                            horse_counter += 1
                            
                            if 'momentum_flavor=effective-clip' in content:
                                self.optimizer_labels[filepath] = f'{horse_name} (Tanea-eff-clip-g3_{g3_val})'
                            elif 'momentum_flavor=mk3' in content:
                                self.optimizer_labels[filepath] = f'{horse_name} (Tanea-mk3-g3_{g3_val})'
                            else:
                                self.optimizer_labels[filepath] = f'{horse_name} (Tanea-unknown-g3_{g3_val})'
                    except:
                        horse_name = horse_names[horse_counter % len(horse_names)]
                        horse_counter += 1
                        self.optimizer_labels[filepath] = f'{horse_name} (Unknown-g3_{g3_val})'
                else:
                    # Fallback for unknown tanea format
                    horse_name = horse_names[horse_counter % len(horse_names)]
                    horse_counter += 1
                    self.optimizer_labels[filepath] = f'{horse_name} (Tanea-unknown)'
            else:
                # Unknown optimizer type
                horse_name = horse_names[horse_counter % len(horse_names)]
                horse_counter += 1
                self.optimizer_labels[filepath] = f'{horse_name} (Unknown)'
        
        # Initialize validation loss and numpy arrays for all files
        for filepath in err_files:
            if filepath in self.optimizer_labels:
                optimizer = self.optimizer_labels[filepath]
                
                # Extract max steps and initialize numpy arrays with 4x capacity
                max_steps = self.extract_max_steps(filepath)
                array_size = max_steps * 4  # 4x the expected max steps for safety
                self.data[optimizer]['max_steps'] = array_size
                self.data[optimizer]['iterations'] = np.full(array_size + 1, -1, dtype=np.int32)  # -1 indicates unused
                self.data[optimizer]['losses'] = np.full(array_size + 1, np.nan, dtype=np.float32)
                self.data[optimizer]['current_idx'] = 0
                
                # Extract validation loss history (last N points)
                val_loss_history = self.extract_val_loss_history(filepath)
                self.data[optimizer]['val_loss_history'] = val_loss_history
                
                val_loss = self.extract_last_val_loss(filepath)
                if val_loss is not None:
                    self.data[optimizer]['val_loss'] = val_loss
                    
                # Get current iteration from recent lines
                current_iter = self.extract_current_iteration(filepath)
                if current_iter is not None:
                    # Set current_idx to 1 and store the current iteration
                    self.data[optimizer]['iterations'][0] = current_iter
                    self.data[optimizer]['current_idx'] = 1
    
    def extract_max_steps(self, filepath):
        """Extract maximum steps from training output"""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    # Look for training progress lines with total steps
                    match = re.search(r'Training:\s*\d+%\|[^|]*\|\s*\d+/(\d+)', line)
                    if match:
                        return int(match.group(1))
        except Exception:
            pass
        return 96000  # Default fallback
    
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
    
    def extract_val_loss_history(self, filepath):
        """Extract validation loss history with actual iterations from end of file"""
        val_loss_history = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
                # Work backwards from the end of the file to find validation losses
                # This ensures we get the most recent validation losses with proper iterations
                recent_val_losses = []
                
                for line in reversed(lines):
                    # Look for validation loss in the line
                    val_match = re.search(r'Val Loss:\s*([0-9]+\.[0-9]+)', line)
                    if val_match and len(recent_val_losses) < self.val_history_limit:
                        val_loss = float(val_match.group(1))
                        
                        # Look for iteration in nearby lines (search a few lines around this one)
                        # Find the index of this line
                        line_idx = len(lines) - 1 - lines[::-1].index(line)
                        
                        # Search for iteration in this line and nearby lines
                        iteration = None
                        search_range = range(max(0, line_idx - 5), min(len(lines), line_idx + 5))
                        
                        for search_idx in search_range:
                            search_line = lines[search_idx]
                            # Look for training progress patterns
                            iter_match = re.search(r'Training:\s*\d+%\|[^|]*\|\s*(\d+)/\d+', search_line)
                            if iter_match:
                                iteration = int(iter_match.group(1))
                                break
                        
                        if iteration is not None:
                            recent_val_losses.append((iteration, val_loss))
                
                # Reverse to get chronological order (oldest to newest)
                val_loss_history = list(reversed(recent_val_losses))
                
        except Exception:
            pass
        return val_loss_history
    
    def extract_last_val_loss(self, filepath):
        """Extract the last validation loss from a file by searching for 'Val Loss: X.X' pattern"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                # Search for all instances of "Val Loss: [number]"
                val_loss_matches = re.findall(r'Val Loss:\s*([0-9]+\.[0-9]+)', content)
                if val_loss_matches:
                    return float(val_loss_matches[-1])  # Return the last one found
        except Exception as e:
            pass
        return None
    
    def extract_current_iteration(self, filepath):
        """Extract the current iteration from the end of the file efficiently"""
        try:
            with open(filepath, 'r') as f:
                # Read last 8KB to get recent training progress
                f.seek(0, 2)  # Go to end
                file_size = f.tell()
                read_size = min(8192, file_size)  # Read last 8KB or entire file
                f.seek(file_size - read_size)
                
                lines = f.read().split('\n')
                
                # Search backwards through recent lines for training progress
                for line in reversed(lines):
                    if 'Training:' in line:
                        iter_match = re.search(r'Training:\s*\d+%\|[^|]*\|\s*(\d+)/\d+', line)
                        if iter_match:
                            return int(iter_match.group(1))
        except Exception:
            pass
        return None
    
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
                            idx = self.data[optimizer]['current_idx']
                            max_steps = self.data[optimizer]['max_steps']
                            
                            # Store data in numpy arrays if we have space
                            if idx < max_steps:
                                self.data[optimizer]['iterations'][idx] = iteration
                                self.data[optimizer]['losses'][idx] = loss
                                self.data[optimizer]['current_idx'] = idx + 1
                                new_data_found = True
                    
                    # Update validation loss if new data was found
                    if new_data_found:
                        optimizer = self.optimizer_labels[filepath]
                        # Update latest validation loss
                        val_loss = self.extract_last_val_loss(filepath)
                        if val_loss is not None:
                            # Update the current validation loss
                            old_val_loss = self.data[optimizer]['val_loss']
                            self.data[optimizer]['val_loss'] = val_loss
                            
                            # If validation loss changed, add to history stack
                            if old_val_loss != val_loss:
                                current_idx = self.data[optimizer]['current_idx']
                                if current_idx > 0:
                                    latest_iter = self.data[optimizer]['iterations'][current_idx-1]
                                    
                                    # Add to history (keep only last N)
                                    history = self.data[optimizer]['val_loss_history']
                                    history.append((latest_iter, val_loss))
                                    if len(history) > self.val_history_limit:
                                        history.pop(0)  # Remove oldest
                            
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
    
    def get_color_for_optimizer(self, optimizer_index):
        """Get rich color name cycling through available colors"""
        colors = ['blue', 'red', 'green', 'yellow', 'magenta', 'cyan', 'white', 'bright_blue', 'bright_red', 'bright_green', 'bright_yellow', 'bright_magenta', 'bright_cyan']
        return colors[optimizer_index % len(colors)]
    
    
    
    def create_scatter_plot(self):
        """Create ASCII scatter plot of iteration vs validation loss"""
        if not self.data:
            return Text("No data available yet")
        
        # Create rich Text object for the plot
        plot_text = Text()
        
        # Get optimizer ordering consistent with status display (by jobID)
        optimizer_filepath_pairs = []
        for filepath, optimizer in self.optimizer_labels.items():
            val_loss_history = self.data[optimizer]['val_loss_history']
            if val_loss_history:  # Has validation loss data
                optimizer_filepath_pairs.append((optimizer, filepath))
        
        # Sort by jobID to match status display ordering
        optimizer_filepath_pairs.sort(key=lambda x: self.file_info[x[1]]['jobid'])
        
        # Collect all validation loss points for plotting
        all_plot_points = []
        letter_map = {}
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letter_idx = 0
        optimizer_index = 0
        
        for optimizer, filepath in optimizer_filepath_pairs:
            val_loss_history = self.data[optimizer]['val_loss_history']
            
            if val_loss_history:
                # Assign letter to optimizer
                if optimizer not in letter_map:
                    letter_map[optimizer] = letters[letter_idx % len(letters)]
                    letter_idx += 1
                
                # Add all validation loss points from history
                for i, (iteration, val_loss) in enumerate(val_loss_history):
                    is_latest = (i == len(val_loss_history) - 1)  # Last point is the latest
                    all_plot_points.append({
                        'optimizer': optimizer,
                        'iteration': iteration,
                        'val_loss': val_loss,
                        'letter': letter_map[optimizer],
                        'color': self.get_color_for_optimizer(optimizer_index),
                        'is_latest': is_latest
                    })
                
                optimizer_index += 1
        
        if not all_plot_points:
            return Text("No validation loss data available yet")
        
        # Create scatter plot
        width = 120
        height = 20
        
        # Find ranges for iteration
        all_current_iters = [point['iteration'] for point in all_plot_points]
        max_iter = max(all_current_iters)
        min_iter = min(all_current_iters) if len(all_current_iters) > 1 else max_iter - 1000
        
        # Find range for validation loss
        all_val_losses = [point['val_loss'] for point in all_plot_points]
        min_val_loss = min(all_val_losses)
        max_val_loss = max(all_val_losses)
        
        # Create plot grid
        plot_grid = [[{'char': ' ', 'color': 'white'} for _ in range(width)] for _ in range(height)]
        
        # Plot each validation loss point
        for point in all_plot_points:
            iteration = point['iteration']
            val_loss = point['val_loss']
            
            # Convert to plot coordinates
            if max_iter > min_iter:
                x = int((iteration - min_iter) / (max_iter - min_iter) * (width - 1))
            else:
                x = width // 2
                
            if max_val_loss > min_val_loss:
                y = int((max_val_loss - val_loss) / (max_val_loss - min_val_loss) * (height - 1))
            else:
                y = height // 2
                
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            
            # Plot the point (use uppercase for latest, lowercase for historical)
            char = point['letter'] if point['is_latest'] else point['letter'].lower()
            plot_grid[y][x] = {'char': char, 'color': point['color']}
        
        # Create plot using rich Text
        plot_text.append("Validation Loss vs Iteration (scatter plot)\n")
        plot_text.append(f"Max Val Loss: {max_val_loss:.3f}\n")
        
        for row in plot_grid:
            plot_text.append("|")
            for cell in row:
                if cell['char'] != ' ':
                    plot_text.append(cell['char'], style=cell['color'])
                else:
                    plot_text.append(cell['char'])
            plot_text.append("|\n")
        
        plot_text.append(f"Min Val Loss: {min_val_loss:.3f}\n")
        plot_text.append(f"Iterations: {min_iter:.0f} to {max_iter}\n")
        
        return plot_text
    
    def create_status_display(self):
        """Create status display as rich Text"""
        status_text = Text()
        
        status_text.append("=" * 80 + "\n")
        status_text.append(f"TRAINING MONITOR - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        status_text.append("=" * 80 + "\n")
        
        # Print stats for each optimizer, ordered by jobID
        # Create list of (optimizer, filepath) pairs to get jobID ordering
        optimizer_filepath_pairs = []
        for filepath, optimizer in self.optimizer_labels.items():
            if self.data[optimizer]['current_idx'] > 0:  # Has data
                optimizer_filepath_pairs.append((optimizer, filepath))
        
        # Sort by jobID
        optimizer_filepath_pairs.sort(key=lambda x: self.file_info[x[1]]['jobid'])
        
        # Create letter assignment for the optimizers (same logic as scatter plot)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letter_idx = 0
        optimizer_index = 0
        
        for optimizer, filepath in optimizer_filepath_pairs:
            horse_name = optimizer.split('(')[0].strip()
            filename_stripped = self.file_info[filepath]['filename_stripped']
            current_idx = self.data[optimizer]['current_idx']
            latest_iter = self.data[optimizer]['iterations'][current_idx-1] if current_idx > 0 else 0
            val_loss = self.data[optimizer]['val_loss']
            
            # Assign letter and color
            letter = letters[letter_idx % len(letters)]
            color = self.get_color_for_optimizer(optimizer_index)
            letter_idx += 1
            optimizer_index += 1
            
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            
            # Add colored letter next to horse name
            status_text.append(letter, style=color)
            status_text.append(f" {horse_name} ({filename_stripped}): iter={latest_iter:6d}, Val={val_loss_str}\n")
        
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
    parser = argparse.ArgumentParser(description='Monitor training progress from .err files')
    parser.add_argument('results_dir', nargs='?', help='Directory containing .err files')
    parser.add_argument('--val-history-limit', type=int, default=10, 
                        help='Number of validation loss points to keep in history (default: 10)')
    
    args = parser.parse_args()
    
    if args.results_dir:
        results_dir = args.results_dir
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
    
    monitor = TrainingMonitor(results_dir, val_history_limit=args.val_history_limit)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
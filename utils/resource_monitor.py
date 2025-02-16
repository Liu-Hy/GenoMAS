"""Resource monitoring utility for tracking CPU, memory, and GPU usage.

This module provides real-time monitoring of system resources used by the current process,
including CPU usage, memory consumption, and GPU utilization (if NVIDIA GPUs are available).
Resources are monitored in a separate daemon thread that automatically terminates with the main process.
"""

import os
import time
import psutil
import threading
from datetime import datetime
try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

class ResourceMonitor:
    """Monitor system resources including CPU, memory, and GPU usage."""
    
    def __init__(self, log_interval: int = 60, log_file: str = None):
        """Initialize the resource monitor.
        
        Args:
            log_interval: How often to log metrics (in seconds)
            log_file: Where to save the metrics log. If None, only print to stdout
        """
        self.log_interval = max(1, log_interval)  # Ensure positive interval
        self.log_file = log_file
        self.running = False
        self.start_time = None
        
        # Track peak usage
        self.peak_metrics = {
            'cpu_percent': 0.0,
            'memory_gb': 0.0
        }
        
        # Initialize GPU monitoring
        self.gpu_count = 0
        if NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(self.gpu_count):
                    self.peak_metrics[f'gpu_{i}_memory_gb'] = 0.0
                    self.peak_metrics[f'gpu_{i}_util'] = 0.0
            except Exception as e:
                print(f"Warning: Failed to initialize NVIDIA monitoring: {str(e)}")
                self.gpu_count = 0
            
    def _get_gpu_metrics(self):
        """Get GPU memory usage and utilization for all available GPUs."""
        if not self.gpu_count:
            return {}
        
        gpu_metrics = {}
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                memory_gb = info.used / (1024 ** 3)  # More readable than /1024/1024/1024
                memory_percent = (info.used / info.total) * 100
                gpu_util = utilization.gpu
                
                # Update peak values
                self.peak_metrics[f'gpu_{i}_memory_gb'] = max(
                    self.peak_metrics[f'gpu_{i}_memory_gb'], 
                    memory_gb
                )
                self.peak_metrics[f'gpu_{i}_util'] = max(
                    self.peak_metrics[f'gpu_{i}_util'], 
                    gpu_util
                )
                
                gpu_metrics[f'gpu_{i}'] = {
                    'memory_gb': memory_gb,
                    'memory_total_gb': info.total / (1024 ** 3),
                    'memory_percent': memory_percent,
                    'utilization': gpu_util
                }
            except Exception as e:
                print(f"Warning: Failed to get metrics for GPU {i}: {str(e)}")
                continue
                
        return gpu_metrics
    
    def _get_system_metrics(self):
        """Get CPU usage and process memory for the current process."""
        try:
            process = psutil.Process(os.getpid())
            
            # Calculate current values
            cpu_percent = process.cpu_percent()
            memory_gb = process.memory_info().rss / (1024 ** 3)
            
            # Update peak values
            self.peak_metrics['cpu_percent'] = max(
                self.peak_metrics['cpu_percent'], 
                cpu_percent
            )
            self.peak_metrics['memory_gb'] = max(
                self.peak_metrics['memory_gb'], 
                memory_gb
            )
            
            return {
                'cpu_percent': cpu_percent,
                'memory_gb': memory_gb
            }
        except Exception as e:
            print(f"Warning: Failed to get system metrics: {str(e)}")
            return {
                'cpu_percent': 0.0,
                'memory_gb': 0.0
            }
        
    def _monitor_loop(self):
        """Main monitoring loop that periodically collects and logs resource usage."""
        while self.running:
            try:
                current_time = datetime.now()
                elapsed_time = time.time() - self.start_time
                
                sys_metrics = self._get_system_metrics()
                gpu_metrics = self._get_gpu_metrics()
                
                # Format output
                output = []
                output.append(f"\n=== Resource Usage at {current_time.strftime('%Y-%m-%d %H:%M:%S')} (Elapsed: {elapsed_time:.1f}s) ===")
                
                # Current Usage - combine CPU and memory on one line
                output.append(f"Current: CPU: {sys_metrics['cpu_percent']:.1f}%, Memory: {sys_metrics['memory_gb']:.1f}GB")
                
                # GPU Resources - one line per GPU
                if self.gpu_count > 0:
                    for i in range(self.gpu_count):
                        if f'gpu_{i}' in gpu_metrics:  # Check if GPU metrics were successfully collected
                            gpu = gpu_metrics[f'gpu_{i}']
                            output.append(
                                f"GPU {i}: {gpu['memory_gb']:.1f}GB/{gpu['memory_total_gb']:.1f}GB "
                                f"({gpu['memory_percent']:.1f}%), Util: {gpu['utilization']}%"
                            )
                
                # Peak Usage - combine all peaks on one line
                peaks = [
                    f"CPU: {self.peak_metrics['cpu_percent']:.1f}%",
                    f"Memory: {self.peak_metrics['memory_gb']:.1f}GB"
                ]
                
                if self.gpu_count > 0:
                    for i in range(self.gpu_count):
                        if f'gpu_{i}_memory_gb' in self.peak_metrics:
                            peaks.append(
                                f"GPU {i}: {self.peak_metrics[f'gpu_{i}_memory_gb']:.1f}GB/"
                                f"{self.peak_metrics[f'gpu_{i}_util']}%"
                            )
                
                output.append("Peak: " + ", ".join(peaks))
                
                output_str = '\n'.join(output)
                print(output_str)
                
                if self.log_file:
                    try:
                        with open(self.log_file, 'a') as f:
                            f.write(output_str + '\n')
                    except Exception as e:
                        print(f"Warning: Failed to write to log file: {str(e)}")
                
            except Exception as e:
                print(f"Error in resource monitoring: {str(e)}")
            
            time.sleep(self.log_interval)
    
    def start(self):
        """Start the resource monitoring in a daemon thread."""
        if not self.running:
            self.running = True
            self.start_time = time.time()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop(self):
        """Stop the resource monitoring.
        Note: This is optional since the daemon thread will automatically terminate
        when the main program exits.
        """
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=self.log_interval)

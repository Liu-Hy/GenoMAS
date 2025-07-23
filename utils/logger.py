import logging
import os
import re
import time
import threading
from typing import Optional


class Logger(logging.Logger):
    """
    Enhanced logger with persistent cumulative statistics across process restarts.
    
    Note: This implementation is designed for single-process use. For multi-process
    scenarios, additional synchronization mechanisms would be needed to prevent
    concurrent write conflicts.
    """
    def __init__(self, log_file: Optional[str] = None, name: Optional[str] = "Script", max_msg_length: int = 5000):
        super().__init__(name=name)
        self.setLevel(logging.INFO)  # Ensure info() messages are logged
        self.start_time = time.time()
        self.api_calls = []
        self.cost_tracking_enabled = True  # Flag to indicate if cost tracking is possible
        self.max_msg_length = max_msg_length  # Set this before any logging calls
        
        # Initialize cumulative statistics
        self.cumulative_duration = 0.0
        self.cumulative_api_duration = 0.0
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        self.cumulative_cost = 0.0
        
        # Don't add timestamp to log file name
        self.log_file = log_file

        # Parse existing log file if it exists
        if self.log_file and os.path.exists(self.log_file):
            self._parse_existing_log()
        
        # Set up handler based on whether log_file is provided
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # Use append mode with delayed opening for thread safety
            handler = ThreadSafeFileHandler(self.log_file, mode='a', delay=True)
            handler.baseFilename = os.path.abspath(self.log_file)  # Store absolute path for comparison
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        # Guard against duplicate handlers
        if self.log_file:
            # Check if a handler for this file already exists
            existing_handler = any(
                (isinstance(h, logging.FileHandler) or isinstance(h, ThreadSafeFileHandler)) and 
                getattr(h, 'baseFilename', None) == os.path.abspath(self.log_file)
                for h in self.handlers
            )
            if not existing_handler:
                self.addHandler(handler)
        else:
            self.addHandler(handler)

        # Initialize the log with a header if it's a new log file
        if not (self.log_file and os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0):
            self.info("=== Log started ===")
        else:
            self.info("=== Log resumed ===")
            # Log the loaded cumulative statistics
            if self.cumulative_api_duration > 0:
                self.info(f"Loaded cumulative statistics - Duration: {self.cumulative_duration:.2f}s, "
                         f"API Duration: {self.cumulative_api_duration:.2f}s, "
                         f"Input Tokens: {self.cumulative_input_tokens}, "
                         f"Output Tokens: {self.cumulative_output_tokens}, "
                         f"Cost: ${self.cumulative_cost:.2f}")

    def _truncate_message(self, message: str) -> str:
        """Truncate message if it exceeds max_msg_length"""
        msg = str(message)
        if len(msg) > self.max_msg_length:
            return msg[:self.max_msg_length] + "... [truncated]"
        return msg

    def debug(self, msg, *args, **kwargs):
        super().debug(self._truncate_message(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        super().info(self._truncate_message(msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        super().warning(self._truncate_message(msg), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        super().error(self._truncate_message(msg), *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        super().critical(self._truncate_message(msg), *args, **kwargs)
    
    def _parse_existing_log(self):
        """
        Parse existing log file to extract cumulative statistics.
        Uses a memory-efficient approach by reading from the end of the file.
        """
        try:
            # Define patterns to search for
            cumulative_pattern = re.compile(
                r'Cumulative Stats - Duration: ([\d.]+)s, API Duration: ([\d.]+)s, '
                r'Input Tokens: (\d+), Output Tokens: (\d+)(?:, Cost: \$([\d.]+))?'
            )
            summary_pattern = re.compile(
                r'Total Duration: ([\d.]+) seconds.*?'
                r'Total API Call Duration: ([\d.]+) seconds.*?'
                r'Total Input Tokens: (\d+).*?'
                r'Total Output Tokens: (\d+).*?'
                r'(?:Total API Cost: \$([\d.]+))?',
                re.DOTALL
            )
            
            # Read file backwards to find the last cumulative stats
            cumulative_match = None
            summary_match = None
            
            with open(self.log_file, 'rb') as f:
                # Start from end of file
                f.seek(0, 2)
                file_size = f.tell()
                
                # Read in chunks from the end
                chunk_size = 8192
                overlap = 1024  # To handle patterns split across chunks
                buffer = ''
                
                pos = file_size
                while pos > 0 and not cumulative_match:
                    # Calculate how much to read
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    
                    # Read chunk and decode
                    chunk = f.read(read_size).decode('utf-8', errors='ignore')
                    buffer = chunk + buffer[:overlap]
                    
                    # Search for cumulative stats pattern
                    for match in cumulative_pattern.finditer(buffer):
                        cumulative_match = match
                    
                    # If no cumulative match, search for summary pattern
                    if not cumulative_match:
                        for match in summary_pattern.finditer(buffer):
                            summary_match = match
                    
                    # Keep only the overlap for next iteration
                    if len(buffer) > overlap:
                        buffer = buffer[-overlap:]
            
            # Process the match
            if cumulative_match:
                self.cumulative_duration = float(cumulative_match.group(1))
                self.cumulative_api_duration = float(cumulative_match.group(2))
                self.cumulative_input_tokens = int(cumulative_match.group(3))
                self.cumulative_output_tokens = int(cumulative_match.group(4))
                # Cost might be missing in cost-tracking-disabled scenarios
                cost_str = cumulative_match.group(5)
                self.cumulative_cost = float(cost_str) if cost_str else 0.0
            elif summary_match:
                self.cumulative_duration = float(summary_match.group(1))
                self.cumulative_api_duration = float(summary_match.group(2))
                self.cumulative_input_tokens = int(summary_match.group(3))
                self.cumulative_output_tokens = int(summary_match.group(4))
                cost_str = summary_match.group(5)
                self.cumulative_cost = float(cost_str) if cost_str else 0.0
                    
            # Update start time to account for the elapsed time
            if self.cumulative_duration > 0:
                self.start_time = time.time() - self.cumulative_duration
                
        except Exception as e:
            # If parsing fails, just start fresh
            self.warning(f"Failed to parse existing log file: {e}")

    def log_api_call(self, duration: float, input_tokens: int, output_tokens: int, cost: float):
        call_data = {
            'timestamp': time.time(),
            'duration': duration,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost
        }
        self.api_calls.append(call_data)
        
        # Update cumulative statistics
        self.cumulative_api_duration += duration
        self.cumulative_input_tokens += input_tokens
        self.cumulative_output_tokens += output_tokens
        
        # Update total elapsed time
        self.cumulative_duration = time.time() - self.start_time

        # If we get a negative cost, disable cost tracking for the entire session
        if cost < 0:
            self.cost_tracking_enabled = False
            self.cumulative_cost = -1.0
        elif self.cost_tracking_enabled:
            self.cumulative_cost += cost

        # Log API call with cumulative statistics
        base_log = f"API Call - Duration: {call_data['duration']:.2f}s, " \
                   f"Input Tokens: {call_data['input_tokens']}, Output Tokens: {call_data['output_tokens']}"

        if self.cost_tracking_enabled:
            base_log += f", Cost: ${cost:.2f}"

        self.info(base_log)
        
        # Log cumulative statistics
        cumulative_log = f"Cumulative Stats - Duration: {self.cumulative_duration:.2f}s, " \
                        f"API Duration: {self.cumulative_api_duration:.2f}s, " \
                        f"Input Tokens: {self.cumulative_input_tokens}, " \
                        f"Output Tokens: {self.cumulative_output_tokens}"
        
        if self.cost_tracking_enabled:
            cumulative_log += f", Cost: ${self.cumulative_cost:.2f}"
            
        self.info(cumulative_log)

    def summarize(self):
        """Write final summary statistics to the log file"""
        # Update total elapsed time one final time
        self.cumulative_duration = time.time() - self.start_time
        
        self.info("\n=== Summary Statistics ===")
        self.info(f"Total Duration: {self.cumulative_duration:.2f} seconds")
        self.info(f"Total API Call Duration: {self.cumulative_api_duration:.2f} seconds")
        self.info(f"Total Input Tokens: {self.cumulative_input_tokens}")
        self.info(f"Total Output Tokens: {self.cumulative_output_tokens}")
        if self.cost_tracking_enabled:
            self.info(f"Total API Cost: ${self.cumulative_cost:.2f}")
        self.info("=== Log ended ===")


class ThreadSafeFileHandler(logging.FileHandler):
    """
    Thread-safe file handler that uses file locking to prevent concurrent write conflicts.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=True):
        super().__init__(filename, mode, encoding, delay)
        self._lock = threading.Lock()
    
    def emit(self, record):
        """
        Emit a record with thread-safe file locking.
        """
        with self._lock:
            try:
                # Use OS-level file locking if available
                if hasattr(os, 'O_EXLOCK'):
                    # BSD/macOS exclusive lock
                    fd = os.open(self.baseFilename, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXLOCK)
                    try:
                        with os.fdopen(fd, 'a', encoding=self.encoding) as f:
                            f.write(self.format(record) + self.terminator)
                            f.flush()
                    finally:
                        # fd is automatically closed by fdopen
                        pass
                else:
                    # Standard approach with thread lock only
                    super().emit(record)
            except Exception:
                self.handleError(record)

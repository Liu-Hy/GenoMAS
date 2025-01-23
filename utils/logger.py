import logging
import os
import time
from typing import Optional


class Logger(logging.Logger):
    def __init__(self, log_file: Optional[str] = None, name: Optional[str] = "Script", max_msg_length: int = 5000):
        super().__init__(name=name)
        self.start_time = time.time()
        self.api_calls = []
        self.total_cost = 0.0  # Track cumulative cost
        self.cost_tracking_enabled = True  # Flag to indicate if cost tracking is possible
        self.max_msg_length = max_msg_length  # Set this before any logging calls

        # Add timestamp to log file name if provided
        if log_file:
            base, ext = os.path.splitext(log_file)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_file = f"{base}_{timestamp}{ext}"
        else:
            self.log_file = None

        # Set up handler based on whether log_file is provided
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if self.log_file:
            handler = logging.FileHandler(self.log_file, mode='w')
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.addHandler(handler)

        # Initialize the log with a header
        self.info("=== Log started ===")
        self.max_msg_length = max_msg_length

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

    def log_api_call(self, duration: float, input_tokens: int, output_tokens: int, cost: float):
        call_data = {
            'timestamp': time.time(),
            'duration': duration,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost
        }
        self.api_calls.append(call_data)

        # If we get a negative cost, disable cost tracking for the entire session
        if cost < 0:
            self.cost_tracking_enabled = False
            self.total_cost = -1.0
        elif self.cost_tracking_enabled:
            self.total_cost += cost

        # Log API call
        base_log = f"API Call - Duration: {call_data['duration']:.2f}s, " \
                   f"Input Tokens: {call_data['input_tokens']}, Output Tokens: {call_data['output_tokens']}"

        if self.cost_tracking_enabled:
            base_log += f", Cost: ${cost:.2f}, Cumulative Cost: ${self.total_cost:.2f}"

        self.info(base_log)

    def save(self):
        """Write final summary statistics to the log file"""
        end_time = time.time()
        total_duration = end_time - self.start_time
        total_api_duration = sum(call['duration'] for call in self.api_calls)
        total_input_tokens = sum(call['input_tokens'] for call in self.api_calls)
        total_output_tokens = sum(call['output_tokens'] for call in self.api_calls)

        self.info("\n=== Summary Statistics ===")
        self.info(f"Total Duration: {total_duration:.2f} seconds")
        self.info(f"Total API Call Duration: {total_api_duration:.2f} seconds")
        self.info(f"Total Input Tokens: {total_input_tokens}")
        self.info(f"Total Output Tokens: {total_output_tokens}")
        if self.cost_tracking_enabled:
            self.info(f"Total API Cost: ${self.total_cost:.2f}")
        self.info("=== Log ended ===")

import argparse

# Global constants
GLOBAL_MAX_TIME = 500000.0


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Gene expression data analysis with LLM-based agents.")
    parser.add_argument('--max_rounds', type=int, default=2, help='Maximum number of revision rounds.')
    parser.add_argument('--de', type=lambda x: (str(x).lower() == 'true'), default=True, help='Include domain expert.')
    parser.add_argument('--cs', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use code snippet.')
    parser.add_argument('--version', type=str, required=True, help='Version string for the current run of experiment.')
    parser.add_argument('--model', type=str, required=True, help='Name of LLM.')
    parser.add_argument('--provider', type=str, default='none',
                        choices=['none', 'openai', 'anthropic', 'google', 'meta', 'deepseek', 'novita'],
                        help='Provider of LLM. Use "none" to auto-detect from model name.')
    parser.add_argument('--api', type=int, default=None,
                        help='Index of API configuration to use (1-based). If not provided, uses default API keys.')
    parser.add_argument('--use-api', action='store_true',
                        help='Use API service for open source models (e.g., Novita for Llama)')
    parser.add_argument('--max-retract', type=int, default=1,
                        help='Maximum number of times allowed to retract to previous steps')
    parser.add_argument('--plan', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Enable context-aware planning. Use "true" or "false".')
    parser.add_argument('--max-time', type=float, default=420,
                        help='Maximum time (in seconds) allowed for the task')
    return parser

import argparse

# Global constants
GLOBAL_MAX_TIME = 500000.0
MAX_STDOUT_LENGTH = 10000  # Maximum characters for stdout before truncation


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Gene expression data analysis with LLM-based agents.")
    parser.add_argument('--max-rounds', type=int, default=2, help='Maximum number of revision rounds.')
    parser.add_argument('--de', type=lambda x: (str(x).lower() == 'true'), default=True, help='Include domain expert.')
    parser.add_argument('--cs', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use code snippet.')
    parser.add_argument('--version', type=str, required=True, help='Version string for the current run of experiment.')
    
    # Default LLM configuration
    parser.add_argument('--model', type=str, required=True, help='Name of LLM.')
    parser.add_argument('--provider', type=str, default='none',
                        choices=['none', 'openai', 'anthropic', 'google', 'ollama', 'deepseek', 'novita'],
                        help='Provider of LLM. Use "none" to auto-detect from model name.')
    parser.add_argument('--api', type=int, default=None,
                        help='Index of API configuration to use (1-based). If not provided, uses default API keys.')
    parser.add_argument('--use-api', action='store_true',
                        help='Use API service for open source models (e.g., Novita for Llama), instead of local deployment')
    
    # Role-specific LLM configurations
    for role in ['pi', 'statistician', 'data-engineer', 'code-reviewer', 'domain-expert', 'planning']:
        parser.add_argument(f'--{role}-model', type=str, default=None,
                            help=f'Model for {role} role. Defaults to --model if not specified.')
        parser.add_argument(f'--{role}-provider', type=str, default='none',
                            choices=['none', 'openai', 'anthropic', 'google', 'ollama', 'deepseek', 'novita'],
                            help=f'Provider for {role} role. Defaults to --provider if not specified.')
        parser.add_argument(f'--{role}-api', type=int, default=None,
                            help=f'API index for {role} role. Defaults to --api if not specified.')
        parser.add_argument(f'--{role}-use-api', action='store_true',
                            help=f'Use API service for {role} role. Defaults to --use-api if not specified.')
    
    parser.add_argument('--max-retract', type=int, default=1,
                        help='Maximum number of times allowed to retract to previous steps')
    parser.add_argument('--plan', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Enable context-aware planning. Use "true" or "false".')
    parser.add_argument('--max-time', type=float, default=420,
                        help='Maximum time (in seconds) allowed for the task')
    parser.add_argument('--thinking', action='store_true',
                        help='Enable Claude extended thinking mode (budget 1024).')
    parser.add_argument('--track-collaboration', action='store_true',
                        help='Enable tracking of agent collaboration patterns for visualization.')
    return parser

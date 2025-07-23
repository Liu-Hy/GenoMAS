import argparse
import copy
import os
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, TYPE_CHECKING

import backoff
import google.generativeai as genai
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from google.api_core import retry as g_retry
from google.generativeai.types import RequestOptions as GoogleRequestOptions, HarmCategory, HarmBlockThreshold
from ollama import AsyncClient as AsyncOllama
from openai import AsyncOpenAI, RequestOptions as OpenAIRequestOptions

from .utils import check_slow_inference, check_recent_openai_model

if TYPE_CHECKING:
    from .logger import Logger

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_PER_RETRY = 30  # seconds
DEFAULT_TIMEOUT_PER_MESSAGE = DEFAULT_MAX_RETRIES * DEFAULT_TIMEOUT_PER_RETRY

"""Model configurations and pricing.

Important:
    - The cost printed in the log is only for reference. Please track actual costs in your LLM provider platform.
    - For inference models (e.g., 'o1'), cost estimation can be much lower than actual cost since hidden thinking tokens
      are charged at output token rates.
    - Pricing information is accurate as of 2025-02-15 but subject to change. Please update monthly.
"""

MODEL_INFO = {
    'openai': {
        'o4-mini-2025-04-16': {'input_price': 1.1, 'output_price': 4.4},
        'o3-2025-04-16': {'input_price': 2.0, 'output_price': 8.0},
        'o3-mini-2025-01-31': {'input_price': 1.1, 'output_price': 4.4},
        'o1-2024-12-17': {'input_price': 15.0, 'output_price': 60.0},
        'o1-mini-2024-09-12': {'input_price': 1.1, 'output_price': 4.4},
        'gpt-4.1-2025-04-14': {'input_price': 2.0, 'output_price': 8.0},
        'gpt-4.1-mini-2025-04-14': {'input_price': 0.4, 'output_price': 1.6},
        'gpt-4o-2024-11-20': {'input_price': 2.5, 'output_price': 10.0},
        'gpt-4o-mini-2024-07-18': {'input_price': 0.15, 'output_price': 0.60}
    },
    'anthropic': {
        'claude-opus-4-20250514': {'input_price': 15.0, 'output_price': 75.0},
        'claude-sonnet-4-20250514': {'input_price': 3.0, 'output_price': 15.0},
        'claude-3-7-sonnet-20250219': {'input_price': 3.0, 'output_price': 15.0},
        'claude-3-5-sonnet-20241022': {'input_price': 3.0, 'output_price': 15.0},
        'claude-3-5-haiku-20241022': {'input_price': 0.8, 'output_price': 4.0}
    },
    'google': {
        'gemini-2.5-pro': {'input_price': 1.25, 'output_price': 10.0},
        'gemini-2.5-flash': {'input_price': 0.15, 'output_price': 3.5},
        'gemini-2.0-flash-001': {'input_price': 0.1, 'output_price': 0.4},
        'gemini-1.5-pro-002': {'input_price': 1.25, 'output_price': 5.0},
        'gemini-1.5-flash-002': {'input_price': 0.075, 'output_price': 0.30}
    },
    'ollama': {
        'deepseek-r1:671b': {'input_price': 0.0, 'output_price': 0.0, 'size': '671B'},
        'deepseek-r1:70b': {'input_price': 0.0, 'output_price': 0.0, 'size': '70B'},
        'deepseek-r1:32b': {'input_price': 0.0, 'output_price': 0.0, 'size': '32B'},
        'deepseek-v3': {'input_price': 0.0, 'output_price': 0.0, 'size': '671B'},
        'qwen3:235b': {'input_price': 0.0, 'output_price': 0.0, 'size': '235B'},
        'qwen3:32b': {'input_price': 0.0, 'output_price': 0.0, 'size': '32B'},
        'llama4:128x17b': {'input_price': 0.0, 'output_price': 0.0, 'size': '400B'},
        'llama4:16x17b': {'input_price': 0.0, 'output_price': 0.0, 'size': '109B'},
        'llama3.3': {'input_price': 0.0, 'output_price': 0.0, 'size': '70B'},
        'llama3.1': {'input_price': 0.0, 'output_price': 0.0, 'size': '8B'},
        'llama3.2': {'input_price': 0.0, 'output_price': 0.0, 'size': '3B'},
        'llama3.2:1b': {'input_price': 0.0, 'output_price': 0.0, 'size': '1B'}
    },
    'novita': {
        'deepseek-r1:671b': {
            'input_price': 0.70,
            'output_price': 2.50,
            'size': '671B',
            'api_name': 'deepseek/deepseek-r1-0528'
        },
        'deepseek-r1:70b': {
            'input_price': 0.80,
            'output_price': 0.80,
            'size': '70B',
            'api_name': 'deepseek/deepseek-r1-distill-llama-70b'
        },
        'deepseek-r1:32b': {
            'input_price': 0.30,
            'output_price': 0.30,
            'size': '32B',
            'api_name': 'deepseek/deepseek-r1-distill-qwen-32b'
        },
        'deepseek-v3': {
            'input_price': 0.33,
            'output_price': 1.30,
            'size': '671B',
            'api_name': 'deepseek/deepseek-v3-0324'
        },
        'qwen3:235b': {
            'input_price': 0.20,
            'output_price': 0.80,
            'size': '235B',
            'api_name': 'qwen/qwen3-235b-a22b-fp8'
        },
        'qwen3:32b': {
            'input_price': 0.10,
            'output_price': 0.45,
            'size': '32B',
            'api_name': 'qwen/qwen3-32b-fp8'
        },
        'llama4:128x17b': {
            'input_price': 0.17,
            'output_price': 0.85,
            'size': '400B',
            'api_name': 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
        },
        'llama4:16x17b': {
            'input_price': 0.10,
            'output_price': 0.50,
            'size': '109B',
            'api_name': 'meta-llama/llama-4-scout-17b-16e-instruct'
        },
        'llama3.3': {
            'input_price': 0.13,
            'output_price': 0.39,
            'size': '70B',
            'api_name': 'meta-llama/llama-3.3-70b-instruct'
        },
        'llama3.1': {
            'input_price': 0.02,
            'output_price': 0.05,
            'size': '8B',
            'api_name': 'meta-llama/llama-3.1-8b-instruct'
        },
        'llama3.2': {
            'input_price': 0.03,
            'output_price': 0.05,
            'size': '3B',
            'api_name': 'meta-llama/llama-3.2-3b-instruct'
        },
        'llama3.2:1b': {
            'input_price': 0.02,
            'output_price': 0.02,
            'size': '1B',
            'api_name': 'meta-llama/llama-3.2-1b-instruct'
        },
    },
    # Warning: DeepSeek official API has experienced frequent latency issues and internal errors since January 2025.
    'deepseek': {
        'deepseek-v3': {
            'input_price': 0.27,
            'output_price': 1.10,
            'api_name': 'deepseek-chat'
        },
        'deepseek-r1:671b': {
            'input_price': 0.55,
            'output_price': 2.19,
            'api_name': 'deepseek-reasoner'
        }
    }
}


def validate_model(provider: str, model: str, use_api: bool = False) -> str:
    """Validate if the model name is valid for the given provider, or detect provider if 'none'.
    For Llama models, the provider will be determined based on use_api flag.
    
    Args:
        provider: Provider name, or 'none' for auto-detection
        model: Model name
        use_api: If True, use API service (Novita) for Llama models instead of local deployment
    
    Returns:
        str: Validated provider name
    """
    if provider.lower() == 'none':
        is_open_source = any(model in MODEL_INFO[p] for p in ['ollama', 'novita'])
        if is_open_source:
            # Choose default provider for open source models
            return 'novita' if use_api else 'ollama'
        else:
            # Auto-detect provider for proprietary models
            for p, models in MODEL_INFO.items():
                if model in models:
                    return p
            supported_models = "\n".join(
                f"- {p}: {', '.join(models.keys())}"
                for p, models in MODEL_INFO.items()
            )
            raise ValueError(f"Could not detect a provider for model: {model}.\nSupported models:\n{supported_models}")

    # Validate the provider and model
    if provider not in MODEL_INFO:
        raise ValueError(f"Invalid provider: {provider}. Must be one of {list(MODEL_INFO.keys())}")
    if model not in MODEL_INFO[provider]:
        raise ValueError(
            f"Invalid model name for provider {provider}.\n"
            f"Valid models are: {list(MODEL_INFO[provider].keys())}"
        )
    return provider


def calculate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an API call in dollars."""
    try:
        config = MODEL_INFO[provider][model]
        input_cost = (input_tokens / 1_000_000) * config['input_price']
        output_cost = (output_tokens / 1_000_000) * config['output_price']
        return input_cost + output_cost
    except (KeyError, ValueError):
        return -1.0  # Return -1 for unknown models/providers


@dataclass
class ModelConfig:
    model_name: str
    provider: str
    api_key: Optional[str] = None
    timeout_per_retry: Optional[float] = None
    timeout_per_message: Optional[float] = None
    max_retries: Optional[int] = None
    organization: Optional[str] = None
    extra_client_params: Optional[Dict[str, Any]] = None
    extra_message_params: Optional[Dict[str, Any]] = None

    @classmethod
    def create(cls, provider: str, model: str, api_index: Optional[str] = None, args: Optional[Any] = None) -> 'ModelConfig':
        """Factory method to create and validate a ModelConfig instance."""
        load_dotenv()

        # Get API key suffix based on config index
        key_suffix = f"_{api_index}" if api_index is not None else ""

        # Allow more time for reasoning models or models with busy APIs
        scaler = 1.0
        thinking = getattr(args, 'thinking', False) if args else False
        if check_slow_inference(model, thinking):
            if ('deepseek' in model.lower() and '671b' in model.lower()):
                scaler = 5.0
            elif ('gemini' in model.lower() and 'pro' in model.lower()):
                scaler = 4.0
            elif ('qwen' in model.lower() and '235b' in model.lower()):
                scaler = 4.0
            else:
                scaler = 2.0
        elif ('deepseek' in model.lower() and 'v3' in model.lower()):
            scaler = 2.0
        print(f'Timeout limit x {scaler} for {model}.')

        # Provider-specific configurations
        provider_configs = {
            'openai': {
                'max_retries': 3,
                'timeout_per_retry': 30.0 * scaler,
                'timeout_per_message': 90.0 * scaler,
                'organization': os.getenv(f'OPENAI_ORGANIZATION{key_suffix}'),
                'api_key': os.getenv(f'OPENAI_API_KEY{key_suffix}')
            },
            'anthropic': {
                'max_retries': 3,
                'timeout_per_retry': 30.0 * scaler,
                'timeout_per_message': 90.0 * scaler,
                'api_key': os.getenv(f'ANTHROPIC_API_KEY{key_suffix}')
            },
            'google': {
                'max_retries': 3,
                'timeout_per_retry': 30.0 * scaler,
                'timeout_per_message': 90.0 * scaler,
                'api_key': os.getenv(f'GOOGLE_API_KEY{key_suffix}')
            },
            'ollama': {
                'max_retries': 3,
                'timeout_per_retry': 30.0 * scaler,
                'timeout_per_message': 90.0 * scaler,
                'extra_message_params': {
                    'num_ctx': 20000,  # Set to ensure sufficient context; may be GPU-intensive
                }
            },
            'novita': {
                'max_retries': 3,
                'timeout_per_retry': 30.0 * scaler,
                'timeout_per_message': 90.0 * scaler,
                'api_key': os.getenv(f'NOVITA_API_KEY{key_suffix}')
            },
            'deepseek': {
                'max_retries': 3,
                'timeout_per_retry': 30.0 * scaler,
                'timeout_per_message': 90.0 * scaler,
                'api_key': os.getenv(f'DEEPSEEK_API_KEY{key_suffix}')
            }
        }

        if provider not in provider_configs:
            raise ValueError(f"Configuration not found for provider: {provider}")

        config = provider_configs[provider]

        # For OpenAI, check both API key and organization
        if provider == 'openai':
            if not config['api_key'] or not config['organization']:
                raise ValueError(
                    f"Missing OpenAI configuration for index {api_index}. "
                    f"Please ensure both OPENAI_API_KEY{key_suffix} and OPENAI_ORGANIZATION{key_suffix} "
                    "are set in your .env file"
                )
        elif provider != 'ollama' and not config['api_key']:  # Ollama doesn't need API key
            raise ValueError(
                f"Missing API key for {provider} with API index {api_index}. "
                f"Please ensure {provider.upper()}_API_KEY{key_suffix} is set in your .env file"
            )

        return cls(
            model_name=model,
            provider=provider,
            **config
        )


def get_role_specific_args(args: argparse.Namespace, role: str) -> argparse.Namespace:
    """Create a namespace with role-specific arguments, falling back to defaults.
    
    Args:
        args: Original argument namespace
        role: Role name (e.g., 'pi', 'statistician', 'data-engineer', etc.)
    
    Returns:
        A new namespace with appropriate model/provider/api/use_api values
    """
    role_args = copy.deepcopy(args)
    
    # Check each configuration parameter
    for param in ['model', 'provider', 'api', 'use_api']:
        role_param = f"{role}_{param}".replace('-', '_')
        
        # Get role-specific value if it exists and is not None/False
        role_value = getattr(args, role_param, None)
        
        # For use_api, check if the role-specific flag was actually set
        if param == 'use_api':
            # Only override if the role-specific flag was explicitly set
            if role_param in args.__dict__ and getattr(args, role_param, False):
                setattr(role_args, param, True)
            # Otherwise keep the default use_api value
        else:
            # For other parameters, override if role-specific value is provided
            if role_value is not None:
                setattr(role_args, param, role_value)
    
    return role_args


def get_llm_client(args: argparse.Namespace, logger: Optional['Logger'] = None) -> 'LLMClient':
    """Get LLM client based on provider and model."""
    # Validate model and determine provider if not specified
    provider = validate_model(args.provider, args.model, getattr(args, 'use_api', False))

    # Create and validate configuration
    config = ModelConfig.create(provider, args.model, args.api, args)

    # Inject extended thinking parameters for Claude (Anthropic) models when requested
    if provider == 'anthropic' and getattr(args, 'thinking', False):
        # Ensure a dict exists
        config.extra_message_params = config.extra_message_params or {}
        # Add thinking configuration if not already provided
        config.extra_message_params.setdefault('thinking', {
            'type': 'enabled',
            'budget_tokens': 1024  # Minimum budget per Anthropic docs
        })

    # Map providers to client classes
    clients = {
        'openai': OpenAIClient,
        'anthropic': AnthropicClient,
        'google': GoogleClient,
        'ollama': OllamaClient,
        'novita': NovitaClient,
        'deepseek': DeepSeekClient
    }

    return clients[provider](config, logger)


class LLMClient(ABC):
    def __init__(self, config: ModelConfig, logger: Optional['Logger'] = None):
        self.config = config
        self.timeout_per_retry = config.timeout_per_retry or DEFAULT_TIMEOUT_PER_RETRY
        self.timeout_per_message = config.timeout_per_message or DEFAULT_TIMEOUT_PER_MESSAGE
        self.max_retries = config.max_retries or DEFAULT_MAX_RETRIES
        self._initialize_client()
        self.logger = logger
        self.model_name = config.model_name

    @abstractmethod
    def _initialize_client(self):
        pass

    @abstractmethod
    async def generate_completion(self, messages: List[Dict[str, str]]) -> Any:
        pass

    def _remove_thinking_process(self, content: str) -> str:
        """Remove thinking process from the response content, to make experiments with DeepSeek-R1 consistent with
        others."""
        cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return cleaned_content.strip()

    def _format_response(self, content: str, input_tokens: int, output_tokens: int, raw_response: Any) -> Dict[
        str, Any]:
        """Format response with standardized structure and calculate cost."""
        cleaned_content = self._remove_thinking_process(content)
        cost = calculate_cost(self.config.provider, self.model_name, input_tokens, output_tokens)
        return {
            "content": cleaned_content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost
            },
            "raw_response": raw_response
        }

    def handle_exception(self, e: Exception) -> Dict[str, Any]:
        if isinstance(e, (ValueError, KeyError)):  # Errors that should stop the program
            raise
        provider_name = self.__class__.__name__.replace('Client', '')
        error_msg = f"{type(e).__name__} in {provider_name} API call: {str(e)}\n{traceback.format_exc()}"

        if hasattr(self, 'logger') and self.logger:
            self.logger.error(error_msg)
        else:
            print(error_msg)

        return {
            "content": "",
            "usage": {},
            "raw_response": None
        }


############################################
#               OpenAIClient
############################################

class OpenAIClient(LLMClient):
    def _initialize_client(self):
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization,
            timeout=self.timeout_per_retry,
            max_retries=self.max_retries,
            **(self.config.extra_client_params or {})
        )

    async def generate_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            # The role for system prompt has changed from 'system' to 'developer' in recent OpenAI models.
            if check_recent_openai_model(self.model_name) and messages[0]["role"] == "system":
                if 'o1-mini' in self.model_name.lower():
                    messages[0]["role"] = "assistant"
                else:
                    messages[0]["role"] = "developer"
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **(self.config.extra_message_params or {})
            )
            return self._format_response(
                content=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                raw_response=response
            )
        except Exception as e:
            return self.handle_exception(e)


############################################
#             AnthropicClient
############################################

class AnthropicClient(LLMClient):
    def _initialize_client(self):
        self.client = AsyncAnthropic(
            api_key=self.config.api_key,
            timeout=self.timeout_per_retry,
            max_retries=self.max_retries,
            **(self.config.extra_client_params or {})
        )

    async def generate_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            # Input messages format:
            # [{"role": "system", "content": system_prompt},
            # {"role": "user", "content": prompt}]

            extra_params = self.config.extra_message_params or {}

            # Determine max_tokens: default 2048, but 3072 when extended thinking is enabled
            max_tokens = 3072 if "thinking" in extra_params else 2048

            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=messages[0]["content"],
                messages=messages[1:],
                **extra_params
            )
            # Extract assistant text: in thinking mode, the first block may be "thinking".
            extracted_text = None
            for block in response.content:
                # Anthropic SDK returns StructuredText objects with .type attr
                if getattr(block, 'type', None) == 'text':
                    extracted_text = block.text
                    break
            if extracted_text is None and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                # Fallback to original behavior
                extracted_text = response.content[0].text

            return self._format_response(
                content=extracted_text or "",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                raw_response=response
            )
        except Exception as e:
            return self.handle_exception(e)


############################################
#               GoogleClient
############################################

class GoogleClient(LLMClient):
    def _initialize_client(self):
        genai.configure(api_key=self.config.api_key)

        self.model = None

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    async def generate_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]

            # Instantiate model here to configure system prompt
            if self.model is None:
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_msg,
                    **(self.config.extra_client_params or {})
                )

            request_options = GoogleRequestOptions(
                retry=g_retry.AsyncRetry(
                    initial=8.0,  # Start with a small value
                    multiplier=2.0,  # Double the backoff each time
                    maximum=self.timeout_per_retry,  # But cap each backoff at a value
                    timeout=self.timeout_per_message,  # Overall "retry window"
                ),
                timeout=self.timeout_per_retry  # The single-request timeout (per attempt)
            )

            response = await self.model.generate_content_async(
                user_msg,
                request_options=request_options,
                safety_settings=self.safety_settings,
                **(self.config.extra_message_params or {})
            )
            return self._format_response(
                content=response.text,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
                raw_response=response
            )
        except Exception as e:
            return self.handle_exception(e)


############################################
#               OllamaClient
############################################
def with_backoff(func):
    async def wrapper(self, *args, **kwargs):
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self.max_retries,
            max_time=self.timeout_per_message
        )
        async def wrapped(*_args, **_kwargs):
            return await func(self, *_args, **_kwargs)

        return await wrapped(*args, **kwargs)

    return wrapper


class OllamaClient(LLMClient):
    def _initialize_client(self):
        extra_message_params = self.config.extra_message_params or {}
        self.chat_params = {
            "num_ctx": extra_message_params.pop("num_ctx", 20000),
        }
        self.client = AsyncOllama(**(self.config.extra_client_params or {}))

    @with_backoff
    async def generate_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Set system prompt according to this:
        https://www.reddit.com/r/ollama/comments/1czw7mj/how_to_set_system_prompt_in_ollama/
        """
        try:
            # Acceptable options from here
            # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
            response = await self.client.chat(
                model=self.model_name,
                messages=messages,
                options=self.chat_params,
                **(self.config.extra_message_params or {})
            )
            return self._format_response(
                content=response['message']['content'],
                input_tokens=response["prompt_eval_count"],
                output_tokens=response["eval_count"],
                raw_response=response
            )
        except Exception as e:
            return self.handle_exception(e)


############################################
#              NovitaClient
############################################
class NovitaClient(LLMClient):
    def _initialize_client(self):
        """Initialize the Novita client with AsyncOpenAI."""
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url="https://api.novita.ai/v3/openai",
            timeout=self.timeout_per_retry,
            max_retries=self.max_retries,
            **(self.config.extra_client_params or {})
        )

    async def generate_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate completion using Novita's API."""
        try:
            model_name = MODEL_INFO['novita'][self.model_name]['api_name']
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **self.config.extra_message_params or {}
            )
            return self._format_response(
                content=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                raw_response=response
            )
        except Exception as e:
            return self.handle_exception(e)


############################################
#              DeepSeekClient
############################################

class DeepSeekClient(LLMClient):
    def _initialize_client(self):
        """Initialize the DeepSeek client with AsyncOpenAI."""
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url='https://api.deepseek.com',
            timeout=self.timeout_per_retry,
            max_retries=self.max_retries,
            **(self.config.extra_client_params or {})
        )

    async def generate_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate completion using DeepSeek's API."""
        try:
            model_name = MODEL_INFO['deepseek'][self.model_name]['api_name']
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **self.config.extra_message_params or {}
            )

            return self._format_response(
                content=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                raw_response=response
            )
        except Exception as e:
            return self.handle_exception(e)

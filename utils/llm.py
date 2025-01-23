import argparse
import os
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

import backoff
import google.generativeai as genai
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from google.api_core import retry as g_retry
from google.generativeai.types import RequestOptions, HarmCategory, HarmBlockThreshold
from ollama import AsyncClient as AsyncLlama
from openai import AsyncOpenAI, RequestOptions

DEFAULT_MAX_RETRIES = 4
DEFAULT_TIMEOUT_PER_RETRY = 30  # seconds # 30
DEFAULT_TIMEOUT_PER_MESSAGE = 120  # 60

# Model configurations
MODEL_INFO = {
    'openai': {
        'gpt-4o-2024-11-20': {'input_price': 2.5, 'output_price': 10.0},
        'gpt-4o-mini-2024-07-18': {'input_price': 0.15, 'output_price': 0.60},
        'o1-2024-12-17': {'input_price': 15.0, 'output_price': 60.0},
        'o1-mini-2024-09-12': {'input_price': 3.0, 'output_price': 12.0}
    },
    'anthropic': {
        'claude-3-5-sonnet-20241022': {'input_price': 3.0, 'output_price': 15.0},
        'claude-3-5-haiku-20241022': {'input_price': 0.8, 'output_price': 4.0}
    },
    'google': {
        'gemini-1.5-pro-002': {'input_price': 1.25, 'output_price': 5.0},
        'gemini-1.5-flash-002': {'input_price': 0.075, 'output_price': 0.30}
    },
    'meta': {
        'llama3.1': {'input_price': 0.0, 'output_price': 0.0, 'size': '8B'},
        'llama3.2': {'input_price': 0.0, 'output_price': 0.0, 'size': '3B'},
        'llama3.2:1b': {'input_price': 0.0, 'output_price': 0.0, 'size': '1B'},
        'llama3.3': {'input_price': 0.0, 'output_price': 0.0, 'size': '70B'}
    },
    'novita': {
        'llama3.1': {
            'input_price': 0.0003,
            'output_price': 0.0003,
            'size': '8B',
            'api_name': 'meta-llama/llama-3.1-8b-instruct'
        },
        'llama3.2': {
            'input_price': 0.0002,
            'output_price': 0.0002,
            'size': '3B',
            'api_name': 'meta-llama/llama-3.2-3b-instruct'
        },
        'llama3.2:1b': {
            'input_price': 0.0001,
            'output_price': 0.0001,
            'size': '1B',
            'api_name': 'meta-llama/llama-3.2-1b-instruct'
        },
        'llama3.3': {
            'input_price': 0.001,
            'output_price': 0.001,
            'size': '70B',
            'api_name': 'meta-llama/llama-3.3-70b-instruct'
        }
    },
    'deepseek': {
        'deepseek-chat': {'input_price': 0.14, 'output_price': 0.28}
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
    # Special handling for Llama models
    is_llama = any(model in MODEL_INFO[p] for p in ['meta', 'novita'])
    if is_llama:
        return 'novita' if use_api else 'meta'

    # Auto-detect provider if not specified
    if provider.lower() == 'none':
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
    def create(cls, provider: str, model: str, api_index: Optional[str] = None) -> 'ModelConfig':
        """Factory method to create and validate a ModelConfig instance."""
        load_dotenv()

        # Get API key suffix based on config index
        key_suffix = f"_{api_index}" if api_index is not None else ""

        # Provider-specific configurations
        provider_configs = {
            'openai': {
                'max_retries': 4,
                'timeout_per_retry': 30.0,
                'timeout_per_message': 120.0,
                'organization': os.getenv(f'OPENAI_ORGANIZATION{key_suffix}'),
                'api_key': os.getenv(f'OPENAI_API_KEY{key_suffix}')
            },
            'anthropic': {
                'max_retries': 4,
                'timeout_per_retry': 30.0,
                'timeout_per_message': 120.0,
                'api_key': os.getenv(f'ANTHROPIC_API_KEY{key_suffix}')
            },
            'google': {
                'max_retries': 4,
                'timeout_per_retry': 30.0,
                'timeout_per_message': 120.0,
                'api_key': os.getenv(f'GOOGLE_API_KEY{key_suffix}')
            },
            'meta': {
                'max_retries': 4,
                'timeout_per_retry': 30.0,
                'timeout_per_message': 120.0,
                'extra_message_params': {
                    'num_ctx': 20000,  # Set to ensure sufficient context; may be GPU-intensive
                }
            },
            'novita': {
                'max_retries': 4,
                'timeout_per_retry': 30.0,
                'timeout_per_message': 120.0,
                'api_key': os.getenv(f'NOVITA_API_KEY{key_suffix}')
            },
            'deepseek': {
                'max_retries': 4,
                'timeout_per_retry': 60.0,
                'timeout_per_message': 240.0,
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
        elif provider != 'meta' and not config['api_key']:  # Meta (Llama) doesn't need API key
            raise ValueError(
                f"Missing API key for {provider}. Please set {provider.upper()}_API_KEY{key_suffix} in your .env file")

        return cls(
            model_name=model,
            provider=provider,
            **config
        )

    def validate(self, provider: str, api_index: Optional[str] = None) -> None:
        """Validate the configuration based on provider requirements."""
        key_suffix = f"_{api_index}" if api_index is not None else ""

        if provider == 'openai':
            if not self.api_key or not self.organization:
                raise ValueError(
                    f"Missing OpenAI configuration for index {api_index}. "
                    f"Please ensure both OPENAI_API_KEY{key_suffix} and OPENAI_ORGANIZATION{key_suffix} "
                    "are set in your .env file"
                )
        elif provider != 'meta' and not self.api_key:  # Meta (Llama) doesn't need API key
            raise ValueError(
                f"Missing API key for {provider} with index {api_index}. "
                f"Please ensure {provider.upper()}_API_KEY{key_suffix} is set in your .env file"
            )


def get_llm_client(args: argparse.Namespace, logger: Optional['Logger'] = None) -> 'LLMClient':
    """Get LLM client based on provider and model."""
    # Validate model and determine provider if not specified
    provider = validate_model(args.provider, args.model, getattr(args, 'use_api', False))

    # Create and validate configuration
    config = ModelConfig.create(provider, args.model, args.api)

    # Map providers to client classes
    clients = {
        'openai': OpenAIClient,
        'anthropic': AnthropicClient,
        'google': GoogleClient,
        'meta': MetaClient,
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

    def _format_response(self, content: str, input_tokens: int, output_tokens: int, raw_response: Any) -> Dict[
        str, Any]:
        """Format response with standardized structure and calculate cost."""
        cost = calculate_cost(self.config.provider, self.model_name, input_tokens, output_tokens)
        return {
            "content": content,
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
            self.logger.log_message(error_msg)
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
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=messages[0]["content"],
                messages=messages[1:],
                **(self.config.extra_message_params or {})
            )
            return self._format_response(
                content=response.content[0].text,
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

            request_options = RequestOptions(
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
#               MetaClient
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


class MetaClient(LLMClient):
    def _initialize_client(self):
        extra_message_params = self.config.extra_message_params or {}
        self.chat_params = {
            "num_ctx": extra_message_params.pop("num_ctx", 20000),
        }
        self.client = AsyncLlama(**(self.config.extra_client_params or {}))

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
            response = await self.client.chat.completions.create(
                model=self.model_name,
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

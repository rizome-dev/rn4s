"""
Client for LiteLLM, providing a unified interface to multiple LLM providers.
"""

import os
import anyio
from typing import Dict, List, Any, Optional, Union

from .base_client import BaseClient, Message, CompletionResponse

class LiteLLMClient(BaseClient):
    """
    Client for LiteLLM, providing a unified interface to multiple LLM providers.
    Supports HuggingFace, OpenAI, Anthropic, Claude, OpenRouter, etc.
    
    Access OpenRouter models by prefixing model names with "openrouter/".
    """
    
    def __init__(
        self, 
        model_id: str = "gpt-3.5-turbo", 
        api_key: Optional[str] = None,
        timeout: int = 180,
        max_retries: int = 3,
        max_tokens: int = 2048,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        force_timeout: bool = True,
        litellm_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new LiteLLMClient.
        
        Args:
            model_id: The model ID to use (can include provider prefix like "openrouter/anthropic/claude-3-opus")
            api_key: API key (defaults to provider-specific env vars)
            timeout: Timeout in seconds for API calls
            max_retries: Maximum number of retries for failed API calls
            max_tokens: Maximum tokens to generate
            site_url: Optional site URL for OpenRouter tracking (sets OR_SITE_URL env var)
            app_name: Optional app name for OpenRouter tracking (sets OR_APP_NAME env var)
            force_timeout: Whether to enforce timeout in LiteLLM calls
            litellm_params: Additional parameters to pass to litellm.completion
        """
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.site_url = site_url
        self.app_name = app_name
        self.force_timeout = force_timeout
        self.litellm_params = litellm_params or {}
        self.model_type = "chat"  # For compatibility with DSPy
        
        # Import litellm here to avoid making it a hard dependency
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is not installed. Install it with pip install litellm"
            )
        
        # Configure environment variables for OpenRouter if applicable
        if "openrouter/" in model_id.lower():
            if site_url:
                os.environ["OR_SITE_URL"] = site_url
            if app_name:
                os.environ["OR_APP_NAME"] = app_name
        
        # Set up model-specific API keys if provided
        if api_key:
            if "openrouter/" in model_id.lower():
                os.environ["OPENROUTER_API_KEY"] = api_key
            elif "openai/" in model_id.lower() or model_id.startswith("gpt-"):
                os.environ["OPENAI_API_KEY"] = api_key
            elif "anthropic/" in model_id.lower() or "claude" in model_id.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
        
        print(f"Using LiteLLM with model: {model_id}")
    
    async def generate(
        self, 
        messages: List[Union[Dict[str, str], Message]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> CompletionResponse:
        """
        Generate a completion using LiteLLM's completion function.
        
        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (overrides instance default)
            top_p: Top-p sampling parameter
            stop_sequences: Sequences that will stop generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            CompletionResponse with the generated text
        """
        # Format messages if needed
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                formatted_messages.append({"role": msg.role, "content": msg.content})
            else:
                formatted_messages.append(msg)
        
        # Merge additional parameters with defaults
        params = {
            "model": self.model_id,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": top_p,
            "request_timeout": self.timeout if self.force_timeout else None,
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            params["stop"] = stop_sequences
            
        # Add additional parameters from litellm_params and kwargs
        params.update(self.litellm_params)
        params.update(kwargs)
        
        # Create limiter for retry logic
        limiter = anyio.CapacityLimiter(1)  # Process one request at a time
        
        async with limiter:
            for attempt in range(self.max_retries):
                try:
                    # Use LiteLLM to make the API call
                    # We need to handle both async and sync versions
                    if hasattr(self.litellm, "acompletion"):
                        # Use async version if available
                        result = await self.litellm.acompletion(**params)
                    else:
                        # Fall back to sync version in async context
                        result = await anyio.to_thread.run_sync(
                            lambda: self.litellm.completion(**params)
                        )
                    
                    # Extract response content from the result
                    if hasattr(result, "choices") and len(result.choices) > 0:
                        # Handle object-style response
                        content = result.choices[0].message.content
                    elif isinstance(result, dict):
                        # Handle dictionary-style response
                        if "choices" in result and len(result["choices"]) > 0:
                            message = result["choices"][0].get("message", {})
                            content = message.get("content", "")
                        else:
                            content = "No content returned from API"
                    else:
                        content = str(result)
                    
                    # Extract usage stats
                    usage = {}
                    if hasattr(result, "usage"):
                        usage = {
                            "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(result.usage, "completion_tokens", 0),
                            "total_tokens": getattr(result.usage, "total_tokens", 0)
                        }
                    elif isinstance(result, dict) and "usage" in result:
                        usage = result["usage"]
                    
                    return CompletionResponse(
                        content=content,
                        model_id=self.model_id,
                        usage=usage
                    )
                
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise ValueError(f"LiteLLM failed after {self.max_retries} attempts: {str(e)}")
                    
                    # Exponential backoff
                    backoff = 2 ** attempt
                    print(f"LiteLLM request failed, retrying in {backoff}s: {str(e)}")
                    await anyio.sleep(backoff) 
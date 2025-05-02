"""
Client for OpenRouter API, providing access to many LLM models.
"""

import os
import httpx
import anyio
from typing import Dict, List, Any, Optional, Union

from .base_client import BaseClient, Message, CompletionResponse

class OpenRouterClient(BaseClient):
    """
    Client for OpenRouter, providing access to many LLM models through a unified API.
    Compatible with the OpenAI API format.
    """
    
    def __init__(
        self, 
        model_id: str = "anthropic/claude-3-opus", 
        api_key: Optional[str] = None,
        api_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 180,
        max_retries: int = 3,
        max_tokens: int = 2048,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        Initialize a new OpenRouterClient.
        
        Args:
            model_id: The model ID to use (provider/model format)
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            api_url: Base URL for the API
            timeout: Timeout in seconds for API calls
            max_retries: Maximum number of retries for failed API calls
            max_tokens: Maximum tokens to generate
            site_url: Optional site URL for OpenRouter leaderboard
            site_name: Optional site name for OpenRouter leaderboard
        """
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.site_url = site_url
        self.site_name = site_name
        self.model_type = "chat"  # For compatibility with DSPy
        
        if not self.api_key:
            raise ValueError("No API key provided. Set OPENROUTER_API_KEY environment variable or pass api_key.")
            
        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add optional headers for OpenRouter leaderboard
        if self.site_url:
            self.headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            self.headers["X-Title"] = self.site_name
            
        # API endpoint
        self.api_endpoint = f"{self.api_url}/chat/completions"
        
        print(f"Using OpenRouter API endpoint: {self.api_endpoint}")
    
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
        Generate a completion from the OpenRouter API.
        
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
        
        # Construct the payload (OpenAI-compatible format)
        payload = {
            "model": self.model_id,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": top_p,
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            payload["stop"] = stop_sequences
            
        # Add extra_body parameters if provided
        if "extra_body" in kwargs:
            for key, value in kwargs["extra_body"].items():
                payload[key] = value
        
        # Create limiter for retry logic
        limiter = anyio.CapacityLimiter(1)  # Process one request at a time
        
        async with limiter:
            for attempt in range(self.max_retries):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            self.api_endpoint,
                            headers=self.headers,
                            json=payload
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        # Extract response content (OpenAI format)
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                        else:
                            content = "No content returned from API"
                        
                        # Extract usage stats
                        usage = result.get("usage", {})
                        
                        return CompletionResponse(
                            content=content,
                            model_id=self.model_id,
                            usage=usage
                        )
                
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if attempt == self.max_retries - 1:
                        raise ValueError(f"Failed to generate after {self.max_retries} attempts: {str(e)}")
                    
                    # Exponential backoff
                    backoff = 2 ** attempt
                    print(f"OpenRouter request failed, retrying in {backoff}s: {str(e)}")
                    await anyio.sleep(backoff) 
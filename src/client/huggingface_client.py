"""
Client for Hugging Face Inference API.
"""

import os
import httpx
import anyio
from typing import Dict, List, Any, Optional, Union

from .base_client import BaseClient, Message, CompletionResponse

class HuggingFaceClient(BaseClient):
    """
    Client for interacting with Hugging Face models via the Inference API.
    """
    
    def __init__(
        self, 
        model_id: str = "Qwen/QwQ-32B", 
        api_token: Optional[str] = None,
        api_url: str = "https://api-inference.huggingface.co/models",
        provider: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        max_tokens: int = 1024,
    ):
        """
        Initialize a new HuggingFaceClient.
        
        Args:
            model_id: The model ID to use
            api_token: Hugging Face API token (defaults to HF_TOKEN env var)
            api_url: Base URL for the API
            provider: Optional provider specification
            timeout: Timeout in seconds for API calls
            max_retries: Maximum number of retries for failed API calls
            max_tokens: Maximum tokens to generate
        """
        self.model_id = model_id
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.api_url = api_url
        self.provider = provider
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.model_type = "chat"  # Default model type for DSPy compatibility
        
        if not self.api_token:
            raise ValueError("No API token provided. Set HF_TOKEN environment variable or pass api_token.")
            
        # Set up for async operations
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Handle provider-specific endpoint logic
        if provider == "sambanova":
            # The exact working endpoint for SambaNova as seen in llm_with_code_executor.py
            self.api_endpoint = "https://router.huggingface.co/sambanova/v1/chat/completions"
            self.is_sambanova = True
        elif provider:
            # Generic third‑party provider routed through HF
            self.api_endpoint = f"{self.api_url}/{self.model_id}?provider={provider}"
            self.is_sambanova = False
        else:
            # Default HuggingFace Inference API endpoint
            self.api_endpoint = f"{self.api_url}/{self.model_id}"
            self.is_sambanova = False
            
        print(f"Using HuggingFace API endpoint: {self.api_endpoint}")
    
    async def generate(
        self, 
        messages: List[Union[Dict[str, str], Message]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> CompletionResponse:
        """
        Generate a completion from the model.
        
        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (overrides instance default if provided)
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
        
        # Construct the payload depending on provider
        if self.provider == "sambanova" or getattr(self, "is_sambanova", False):
            # EXACT SambaNova payload format from working example in llm_with_code_executor.py
            payload = {
                "messages": formatted_messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "model": self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id,
            }
            
            # Only add these if they're explicitly provided (don't add with defaults)
            if stop_sequences:
                payload["stop"] = stop_sequences
            
            # Add other kwargs as needed, but don't override existing keys
            if kwargs:
                for k, v in kwargs.items():
                    if k not in payload:
                        payload[k] = v
        else:
            # Standard HF Inference API payload
            payload = {
                "inputs": formatted_messages,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens or self.max_tokens,
                    "top_p": top_p,
                    "return_full_text": False,
                }
            }
            # Add any additional parameters to parameters section
            for key, value in kwargs.items():
                if key not in payload["parameters"]:
                    payload["parameters"][key] = value
            if stop_sequences:
                payload["parameters"]["stop"] = stop_sequences
        
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
                        
                        # Parse the response based on its structure
                        if self.is_sambanova:
                            # SambaNova response has OpenAI-like format
                            if "choices" in result and len(result["choices"]) > 0:
                                content = result["choices"][0]["message"]["content"]
                            else:
                                content = "No content returned from API"
                            
                            # Clean up usage data - keep only fields we need and convert to expected types
                            if "usage" in result:
                                usage = {
                                    "prompt_tokens": int(result["usage"].get("prompt_tokens", 0)),
                                    "completion_tokens": int(result["usage"].get("completion_tokens", 0)),
                                    "total_tokens": int(result["usage"].get("total_tokens", 0))
                                }
                            else:
                                usage = {}
                        elif isinstance(result, list) and len(result) > 0:
                            if "generated_text" in result[0]:
                                content = result[0]["generated_text"]
                            elif "message" in result[0] and "content" in result[0]["message"]:
                                content = result[0]["message"]["content"]
                            else:
                                content = str(result[0])
                        elif isinstance(result, dict):
                            if "generated_text" in result:
                                content = result["generated_text"]
                            elif "message" in result and "content" in result["message"]:
                                content = result["message"]["content"]
                            else:
                                content = str(result)
                        else:
                            content = str(result)
                        
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
                    print(f"Request failed, retrying in {backoff}s: {str(e)}")
                    await anyio.sleep(backoff) 
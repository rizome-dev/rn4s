"""
Base classes for LLM client implementations.
"""

import anyio
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

class Message(BaseModel):
    """Message model for chat interactions."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")

class CompletionResponse(BaseModel):
    """Response from a model completion."""
    content: str = Field(..., description="Generated text from the model")
    model_id: str = Field(..., description="Model ID used for generation")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Token usage statistics")

class BaseClient:
    """
    Base class for all LLM clients.
    """
    
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
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop_sequences: Sequences that will stop generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            CompletionResponse with the generated text
        """
        raise NotImplementedError("Subclasses must implement generate method")
    
    async def chat(self, 
        messages: List[Union[Dict[str, str], Message]],
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Simplified chat interface that returns just the text.
        
        Args:
            messages: List of messages in the conversation
            temperature: Temperature for generation
            **kwargs: Additional parameters to pass to generate
            
        Returns:
            String containing the generated text
        """
        response = await self.generate(messages, temperature=temperature, **kwargs)
        return response.content
        
    def __call__(self, messages: List[Union[Dict[str, str], Message]], temperature: float = 0.7, **kwargs) -> str:
        """
        Synchronous interface for compatibility with existing code.
        
        Args:
            messages: List of messages
            temperature: Temperature for generation
            **kwargs: Additional parameters to pass to generate
            
        Returns:
            String response from the model
        """
        # Use anyio to run the async function in a synchronous context
        return anyio.run(self.chat, messages, temperature=temperature, **kwargs) 
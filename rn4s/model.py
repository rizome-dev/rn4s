import os
from typing import List, Optional, Any, Dict, Union

from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient

class JudgeEvaluation(BaseModel):
    """A model to store judge evaluations of agent responses."""
    step_number: int = Field(..., description="Step number in the agent execution")
    evaluation_score: float = Field(..., description="Evaluation score from 0.0 to 1.0")
    feedback: str = Field(..., description="Feedback from the judge about the quality of the response")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the evaluation")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
class JudgeConfig(BaseModel):
    """Configuration for the judge evaluator."""
    prompt_template: str = Field(..., description="Template for the judge prompt")
    criteria: List[str] = Field(default_factory=list, description="Criteria for evaluation")
    weight_quality: float = Field(0.6, description="Weight for quality assessment")
    weight_relevance: float = Field(0.4, description="Weight for relevance assessment")
    
class ProgramThoughtConfig(BaseModel):
    """Configuration for the Program of Thought module."""
    max_iterations: int = Field(3, description="Maximum number of iterations for program refinement")
    language: str = Field("python", description="Programming language to use")
    execution_timeout: Optional[int] = Field(None, description="Timeout for code execution in seconds")
    allowed_imports: List[str] = Field(default_factory=list, description="List of allowed import modules")


class ChatMessage(BaseModel):
    """A class representing a chat message."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


class QwenModel:
    """A wrapper for the Qwen/QwQ model using HuggingFace's InferenceClient."""
    
    def __init__(
        self, 
        model_name: str = "Qwen/QwQ-32B", 
        provider: str = "sambanova",
        api_key: Optional[str] = None,
        max_tokens: int = 512
    ):
        """
        Initialize the QwenModel.
        
        Args:
            model_name: Name of the model to use
            provider: Provider for the model (e.g., sambanova)
            api_key: HuggingFace API key (defaults to HF_TOKEN env var if None)
            max_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.max_tokens = max_tokens
        
        # Set attributes for DSPy integration
        self.model_id = f"{provider}/{model_name}"
        self.model_type = "chat"
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in HF_TOKEN environment variable")
        
        self.client = InferenceClient(
            provider=provider,
            api_key=self.api_key,
        )
    
    def __call__(self, messages: List[Union[Dict[str, str], ChatMessage]]) -> Any:
        """
        Call the model with the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys or ChatMessage objects
            
        Returns:
            CompletionResponse with content attribute containing the model's response
        """
        try:
            # Convert ChatMessage objects to dicts if needed
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    formatted_messages.append({"role": msg.role, "content": msg.content})
                else:
                    formatted_messages.append(msg)
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                max_tokens=self.max_tokens,
            )
            
            # Return a simple wrapper that mimics the Response object
            if completion and hasattr(completion, 'choices') and len(completion.choices) > 0:
                return CompletionResponse(completion.choices[0].message.content)
            else:
                return CompletionResponse("I'm having trouble generating a response at the moment.")
        except Exception as e:
            import traceback
            print(f"Error calling model: {str(e)}\n{traceback.format_exc()}")
            return CompletionResponse(f"Error generating response: {str(e)}")


class CompletionResponse:
    """A simple wrapper around the completion response."""
    
    def __init__(self, content: str):
        """Initialize with content."""
        self.content = content 
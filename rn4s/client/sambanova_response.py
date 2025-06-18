"""
Response models for SambaNova API responses.
"""

from typing import Optional, Dict, Any, List
import time

class SambanovaResponse:
    """
    Response model for SambaNova API responses.
    Provides a consistent interface for handling responses from SambaNova's API.
    """
    
    def __init__(
        self,
        content: str,
        status_code: int = 200,
        request_id: Optional[str] = None,
        raw_response: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        finish_reason: Optional[str] = None,
        latency: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the response model with the response data.
        
        Args:
            content: The generated text content
            status_code: HTTP status code of the response
            request_id: Unique identifier for the request
            raw_response: The full raw response from the API
            model_id: ID of the model used for generation
            usage: Token usage statistics
            finish_reason: Reason for generation completion (e.g., "stop", "length")
            latency: Time taken for the request in seconds
            metadata: Additional metadata from the response
        """
        self.content = content
        self.status_code = status_code
        self.request_id = request_id
        self.raw_response = raw_response or {}
        self.model_id = model_id
        self.usage = usage or {}
        self.finish_reason = finish_reason
        self.latency = latency
        self.metadata = metadata or {}
        self._start_time = time.time()
        
    @classmethod
    def from_api_response(cls, response: Dict[str, Any], start_time: Optional[float] = None) -> "SambanovaResponse":
        """
        Create a SambanovaResponse instance from the raw API response.
        
        Args:
            response: The raw API response dictionary
            start_time: Optional timestamp when the request was started
            
        Returns:
            A SambanovaResponse instance
        """
        # Calculate latency if start_time was provided
        latency = None
        if start_time is not None:
            latency = time.time() - start_time
            
        # Handle different response formats from SambaNova API
        content = ""
        if "generated_text" in response:
            content = response["generated_text"]
        elif "choices" in response and response["choices"]:
            if "message" in response["choices"][0]:
                content = response["choices"][0]["message"].get("content", "")
            elif "text" in response["choices"][0]:
                content = response["choices"][0].get("text", "")
        
        # Extract metadata from different possible formats
        finish_reason = None
        if "choices" in response and response["choices"]:
            finish_reason = response["choices"][0].get("finish_reason")
        
        # Extract usage information
        usage = None
        if "usage" in response:
            usage = response["usage"]
            
        # Extract request ID
        request_id = response.get("id") or response.get("request_id")
        
        # Extract model ID
        model_id = response.get("model")
        
        return cls(
            content=content,
            status_code=200,  # Assume success here, real status code should be from the HTTP response
            request_id=request_id,
            raw_response=response,
            model_id=model_id,
            usage=usage,
            finish_reason=finish_reason,
            latency=latency
        )
    
    def __str__(self) -> str:
        """String representation of the response, showing primary content."""
        return self.content
    
    def __repr__(self) -> str:
        """Detailed representation of the response object."""
        return (
            f"SambanovaResponse(content='{self.content[:50]}{'...' if len(self.content) > 50 else ''}', "
            f"status_code={self.status_code}, model_id='{self.model_id}', "
            f"finish_reason='{self.finish_reason}', latency={self.latency:.2f}s)"
        )
        
    @property
    def elapsed_time(self) -> float:
        """Calculate the elapsed time since this response object was created."""
        return time.time() - self._start_time 
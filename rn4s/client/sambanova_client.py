"""
SambaNova-specific client implementation.

This module provides a specialized client for SambaNova's models,
which are accessed through the Hugging Face Inference API with specific provider settings.
"""

import logging
from typing import Optional

from .huggingface_client import HuggingFaceClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sambanova_client")

def create_sambanova_client(
    model_id: str, 
    api_token: Optional[str] = None
) -> HuggingFaceClient:
    """
    Create a HuggingFaceClient configured for SambaNova models.
    
    Args:
        model_id: The SambaNova model ID
        api_token: API token (defaults to HF_TOKEN env var)
        
    Returns:
        Configured HuggingFaceClient for SambaNova
    """
    return HuggingFaceClient(
        model_id=model_id,
        api_token=api_token,
        provider="sambanova"
    )

# For backwards compatibility, maintain the SambanovaClient class name
SambanovaClient = HuggingFaceClient 
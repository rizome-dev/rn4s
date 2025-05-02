"""
Core module for custom agent implementation.
"""

# Export client classes from the client directory
from .client import (
    BaseClient,
    HuggingFaceClient,
    OpenRouterClient,
    LiteLLMClient,
    Message,
    CompletionResponse,
    SambanovaClient,
    create_sambanova_client,
    SambanovaResponse
)

# Export classes from tools.py
from .tools import (
    Tool, 
    ToolOutput, 
    ToolParameter, 
    ToolMetadata, 
    tool, 
    FinalAnswer, 
    SearchEngine,
    WebScraper,
    add_final_answer_to_agent,
    get_default_tools
)

# Export classes from execution.py
from .execution import CodeExecutor, CodeExecutionResult

# Export classes from judge.py
from .judge import JudgeEvaluator, JudgeEvaluation, JudgeConfig

# Export classes from reasoning.py
from .reasoning import (
    DSPyReasoningAdapter, 
    ReasoningStep, 
    ReasoningChain
)

# Export classes from agent.py
from .agent import (
    Agent,
    AgentMode,
    ActionStep,
    AgentMemory,
    ToolCall
)

# Default judge criteria and prompt
from .judge import DEFAULT_JUDGE_CRITERIA, DEFAULT_JUDGE_PROMPT

__all__ = [
    # Clients
    "BaseClient",
    "HuggingFaceClient",
    "OpenRouterClient",
    "LiteLLMClient",
    "SambanovaClient",
    "create_sambanova_client",
    "Message",
    "CompletionResponse",
    "SambanovaResponse",
    
    # Tools
    "Tool",
    "ToolOutput",
    "ToolParameter",
    "ToolMetadata",
    "tool",
    "FinalAnswer",
    "SearchEngine",
    "WebScraper",
    "add_final_answer_to_agent",
    "get_default_tools",
    
    # Code Execution
    "CodeExecutor",
    "CodeExecutionResult",
    
    # Judge
    "JudgeEvaluator",
    "JudgeEvaluation",
    "JudgeConfig",
    "DEFAULT_JUDGE_CRITERIA",
    "DEFAULT_JUDGE_PROMPT",
    
    # Reasoning
    "DSPyReasoningAdapter",
    "ReasoningStep",
    "ReasoningChain",
    
    # Agent
    "Agent",
    "AgentMode",
    "ActionStep",
    "AgentMemory",
    "ToolCall"
]

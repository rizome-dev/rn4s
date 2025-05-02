# rn4s - Self-Reinforcing Code Agent Framework

- **Flexible Agent Framework**: Create custom agents with different execution modes (chat, reasoning, tool use, code execution)
- **Self-Evaluation System**: Built-in judge evaluator provides structured feedback and drives agent improvement
- **Dual-Agent Architecture**: CodingAgent implements a two-agent approach (Project Manager + Coder) for high-quality code generation
- **Tool Integration**: Smolagents & Langchain-compatible tool system for web search, code execution, and more
- **Memory Management**: Robust memory system tracks conversation history and context
- **Reasoning Chains**: DSPy-compatible reasoning adapter for step-by-step problem-solving

## 📁 Project Structure

```
src/
├── agent/
│   ├── coding_agent.py   # Dual-agent architecture for code generation
│   └── __init__.py       # Agent module exports
├── core/
│   ├── agent.py          # Base agent framework and execution loop
│   ├── client/           # Model client implementations (HuggingFace, OpenRouter, etc.)
│   ├── execution.py      # Code execution utilities and sandbox
│   ├── formatting.py     # Output formatting and tracing tools
│   ├── judge.py          # Evaluation system for agent steps
│   ├── model.py          # Model interfaces and configuration
│   ├── reasoning.py      # Reasoning components and DSPy adapter
│   ├── tool/             # Tool implementations directory
│   ├── tools.py          # Tool definitions and registration
│   └── __init__.py       # Core module exports
└── __init__.py           # Package exports
```

## 🏁 Getting Started

### Basic Agent Usage

```python
from src.core import Agent, HuggingFaceClient

# Create a client for your preferred model
client = HuggingFaceClient(model_id="your-model", api_token="your-token")

# Initialize an agent
agent = Agent(
    client=client,
    verbose=True,
    max_steps=5,
    mode="code_execution"  # Options: "chat", "reasoning", "tool_use", "code_execution"
)

# Run the agent on a task
result = agent.run("Write a function to calculate Fibonacci numbers")
```

### Using the CodingAgent (Dual-Agent Architecture)

```python
from src.agent import CodingAgent

# Initialize the coding agent with the two-agent architecture
agent = CodingAgent(
    project_manager_api_token="your-sambanova-token",
    openrouter_api_key="your-openrouter-key",
    max_iterations=5,
    verbose=True
)

# Generate code from a natural language description
result = agent.run("Create a Python function that analyzes sentiment in text")
```

## 🧩 Core Components

### Agent Framework (`Agent` class)

The `Agent` class provides the foundation for all agents in Judge, with features including:
- Step-by-step execution with evaluation
- Memory management for context tracking
- Multiple execution modes for different use cases
- Prompt optimization based on feedback
- Rich tracing and output formatting

### Judge Evaluation System (`JudgeEvaluator` class)

The `JudgeEvaluator` provides structured evaluation of agent steps:
- Scores agent responses on customizable criteria (correctness, relevance, code quality, etc.)
- Generates detailed feedback highlighting strengths and weaknesses
- Optimizes prompts based on past evaluations
- Improves agent performance through iterative refinement

### Coding Agent Architecture (`CodingAgent` class)

The `CodingAgent` implements a two-agent approach for high-quality code generation:
1. **Project Manager**: Defines specifications, reviews code, provides feedback
2. **Coder**: Implements code based on specifications

This dual-agent architecture enables:
- Detailed specification development
- Iterative code implementation and review
- Automatic code execution for validation
- Progressive improvement through feedback

## 🛠️ Customization

Judge is designed to be highly customizable:
- Implement custom tools by extending the `Tool` class
- Create custom clients for different LLM providers
- Define your own evaluation criteria for the judge system
- Customize the system prompt and reasoning templates

## 📊 Evaluation and Testing

The Judge framework includes built-in evaluation capabilities:
- Track agent performance metrics across steps
- Capture detailed logs of agent reasoning
- Visualize execution traces with the `AgentTracer`
- Compare results across different model configurations

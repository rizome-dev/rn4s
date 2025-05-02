"""
Custom agent implementation with robust reasoning capabilities.
"""

import re
import json
import asyncio
import inspect
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
import anyio
from pydantic import BaseModel, Field

from .client.huggingface_client import HuggingFaceClient, Message
from .tools import Tool, ToolOutput, get_default_tools
from .reasoning import DSPyReasoningAdapter, ReasoningChain, ReasoningStep
from .execution import CodeExecutor, CodeExecutionResult
from .judge import JudgeEvaluator, JudgeEvaluation, JudgeConfig
from .formatting import AgentTracer, ThinkOutputParser, estimate_tokens

class AgentMode(str, Enum):
    """Execution mode for the agent."""
    CHAT = "chat"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    CODE_EXECUTION = "code_execution"

class ToolCall(BaseModel):
    """A call to a tool made by the agent."""
    tool_name: str = Field(..., description="Name of the tool being called")
    tool_input: Dict[str, Any] = Field(default_factory=dict, description="Input parameters for the tool")
    
class ActionStep(BaseModel):
    """A step in the agent's action sequence."""
    step_number: int = Field(..., description="Sequential number of this step")
    thoughts: Optional[str] = Field(None, description="Agent's internal thoughts for this step")
    action: Optional[ToolCall] = Field(None, description="Tool call for this step")
    code: Optional[str] = Field(None, description="Code executed in this step")
    observation: Optional[str] = Field(None, description="Observation from tool execution or code execution")
    error: Optional[str] = Field(None, description="Error message if any")
    evaluation: Optional[JudgeEvaluation] = Field(None, description="Judge evaluation of this step")
    # New field for raw model output
    raw_output: Optional[str] = Field(None, description="Raw model output before parsing")

class AgentMemory(BaseModel):
    """The agent's memory of past interactions."""
    system_prompt: str = Field("", description="System prompt for the agent")
    task: str = Field("", description="Current task for the agent")
    steps: List[ActionStep] = Field(default_factory=list, description="Steps taken by the agent")
    max_memory_items: int = Field(20, description="Maximum number of memory items to keep")
    evaluations: List[JudgeEvaluation] = Field(default_factory=list, description="Evaluations from judge")
    context: str = Field("", description="Current context of the agent's memory")
    
    def add_step(self, step: ActionStep):
        """
        Add a step to the memory and update context.
        
        Args:
            step: Step to add
        """
        self.steps.append(step)
        
        # If step has evaluation, add to evaluations list
        if step.evaluation:
            self.evaluations.append(step.evaluation)
            
        # Trim if necessary
        if len(self.steps) > self.max_memory_items:
            self.steps = self.steps[-self.max_memory_items:]
            
        # Update context
        self._update_context()
    
    def _update_context(self):
        """Update the current context based on memory contents."""
        context = f"Task: {self.task}\n\n"
        
        for i, step in enumerate(self.steps):
            context += f"Step {i+1}:\n"
            
            if step.thoughts:
                context += f"Thoughts: {step.thoughts}\n"
                
            if step.action:
                context += f"Action: Called tool '{step.action.tool_name}' with parameters {json.dumps(step.action.tool_input)}\n"
                
            if step.code:
                # Truncate code if it's too long
                code_snippet = step.code
                if len(code_snippet) > 500:
                    code_snippet = code_snippet[:500] + "... [truncated]"
                context += f"Code:\n{code_snippet}\n"
                
            if step.observation:
                context += f"Observation: {step.observation}\n"
                
            if step.error:
                context += f"Error: {step.error}\n"
                
            if step.evaluation:
                context += f"Evaluation: {step.evaluation.feedback} (Score: {step.evaluation.evaluation_score})\n"
                
            context += "\n"
            
        self.context = context
    
    def get_context(self) -> str:
        """
        Get the current context from memory.
        
        Returns:
            Context string
        """
        return self.context

class Agent(BaseModel):
    """
    A custom agent using Pydantic for type safety and AnyIO for async operations.
    This implementation is inspired by smolagents for better output formatting and planning.
    """
    
    # Configuration
    model_config = {"arbitrary_types_allowed": True}
    
    # Required inputs
    client: Any = Field(..., description="LLM client for the agent")
    
    # Optional configuration
    tools: List[Tool] = Field(default_factory=list, description="Tools available to the agent")
    system_prompt: str = Field("You are a helpful AI assistant.", description="System prompt for the agent")
    max_steps: int = Field(10, description="Maximum number of steps to take")
    mode: AgentMode = Field(default=AgentMode.CODE_EXECUTION, description="Execution mode for the agent")
    optimize_prompt: bool = Field(True, description="Whether to optimize the prompt after each step")
    planning_interval: int = Field(0, description="How often to run a planning step (0 = never)")
    verbose: bool = Field(True, description="Whether to show detailed output")
    show_thinking: bool = Field(False, description="Whether to show thinking steps (which can be very verbose)")
    
    # Components
    code_executor: Optional[CodeExecutor] = Field(None, description="Code executor for running Python code")
    reasoning_adapter: Optional[DSPyReasoningAdapter] = None
    judge_evaluator: Optional[JudgeEvaluator] = None
    
    # State
    memory: AgentMemory = Field(default_factory=AgentMemory, description="Agent's memory")
    
    # Output formatting and parsing
    tracer: Optional[AgentTracer] = None
    think_parser: Optional[ThinkOutputParser] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set up memory with system prompt
        self.memory.system_prompt = self.system_prompt
        
        # Initialize components if not provided
        if not self.code_executor:
            self.code_executor = CodeExecutor()
            
        if not self.reasoning_adapter:
            self.reasoning_adapter = DSPyReasoningAdapter(self.client)
            
        if not self.judge_evaluator:
            self.judge_evaluator = JudgeEvaluator(self.client)
        
        # Initialize tracer and parser
        if not self.tracer:
            self.tracer = AgentTracer(verbose=self.verbose, show_thinking=self.show_thinking)
            
        if not self.think_parser:
            self.think_parser = ThinkOutputParser()
            
        # If no tools are provided, use default tools
        if not self.tools:
            self.tools = get_default_tools()
        
    async def arun(self, task: str) -> str:
        """
        Run the agent asynchronously on a task.
        
        Args:
            task: The task for the agent to solve
            
        Returns:
            The agent's final answer
        """
        # Initialize the run
        self.memory = AgentMemory(system_prompt=self.system_prompt)
        self.memory.task = task
        
        # Start task in tracer
        self.tracer.start_task(task)
        
        # Generate initial reasoning if in reasoning mode
        if self.mode in [AgentMode.REASONING, AgentMode.CODE_EXECUTION]:
            reasoning = await self._generate_reasoning(task)
            # Enhance system prompt with reasoning
            self.memory.system_prompt = self.reasoning_adapter.add_reasoning_to_system_prompt(
                self.memory.system_prompt, reasoning
            )
        
        # Execute steps
        step_number = 0
        final_answer = None
        
        while step_number < self.max_steps:
            step_number += 1
            
            # Check if we should run a planning step
            if self.planning_interval > 0 and step_number % self.planning_interval == 0:
                await self._run_planning_step(step_number)
            
            # Start step in tracer
            self.tracer.start_step(step_number)
            
            # Execute a step
            result, is_final, token_stats = await self._execute_step(step_number)
            
            # End step in tracer
            self.tracer.end_step(token_stats)
            
            if is_final:
                final_answer = result
                break
        
        # If we ran out of steps, generate a final answer
        if final_answer is None:
            final_answer = await self._generate_final_answer()
        
        # Print final answer
        self.tracer.final_answer(final_answer)
        
        return final_answer
        
    def run(self, task: str) -> str:
        """
        Run the agent synchronously on a task.
        
        Args:
            task: The task for the agent to solve
            
        Returns:
            The agent's final answer
        """
        return anyio.run(self.arun, task)
    
    async def _run_planning_step(self, step_number: int):
        """
        Run a planning step to update goals and strategies.
        
        Args:
            step_number: Current step number
        """
        messages = self._create_messages()
        
        # Add a special planning prompt
        messages.append({
            "role": "user",
            "content": (
                "Let's pause and do some planning. Based on what you've learned so far:\n"
                "1. List 2-3 key facts you've discovered.\n"
                "2. What are your next 1-2 steps to solve this task?\n"
                "3. Is there anything you're missing or need to reconsider?"
            )
        })
        
        # Call the model
        response = await self._call_model(messages)
        
        # Create a planning step
        action_step = ActionStep(step_number=step_number)
        action_step.thoughts = f"Planning step:\n{response}"
        action_step.raw_output = response
        
        # Add to memory
        self.memory.add_step(action_step)
        
        # Display planning in tracer
        self.tracer.add_thinking(response)
        
        return response
    
    async def _execute_step(self, step_number: int) -> Tuple[str, bool, Dict[str, int]]:
        """
        Execute a single step of the agent.
        
        Args:
            step_number: The step number
            
        Returns:
            Tuple of (result, is_final, token_stats) where:
            - result is the step output
            - is_final indicates if this is the final answer
            - token_stats is a dictionary with token count information
        """
        # Create messages for the model
        messages = self._create_messages()
        
        # Estimate input tokens
        input_text = "\n".join([msg["content"] for msg in messages])
        input_tokens = estimate_tokens(input_text)
        
        # Call the model
        response = await self._call_model(messages)
        
        # Estimate output tokens
        output_tokens = estimate_tokens(response)
        
        # Token stats for tracing
        token_stats = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        
        # Parse thinking/reasoning if available
        thoughts = self.think_parser.extract_thoughts(response)
        content = self.think_parser.extract_content(response)
        
        if thoughts:
            self.tracer.add_thinking(thoughts)
        
        if self.mode == AgentMode.CODE_EXECUTION:
            # Extract code and reasoning
            action_step, is_final_answer, code = self._parse_code_response(response, step_number)
            
            # Store raw output and thoughts
            action_step.raw_output = response
            if thoughts and not action_step.thoughts:
                action_step.thoughts = thoughts
            
            if is_final_answer:
                return action_step.thoughts or response, True, token_stats
                
            # Execute the code
            if code:
                action_step.code = code
                self.tracer.add_code(code)
                
                execution_result = await self.code_executor.execute_code(code)
                
                if execution_result.success:
                    # Combine stdout and result
                    result_text = execution_result.stdout
                    if execution_result.result:
                        if result_text:
                            result_text += f"\nResult: {execution_result.result}"
                        else:
                            result_text = f"Result: {execution_result.result}"
                    action_step.observation = result_text
                    self.tracer.add_output(result_text)
                else:
                    # Store error
                    action_step.error = execution_result.error or execution_result.stderr
                    self.tracer.add_error(action_step.error)
                
                # Get judge evaluation
                evaluation = await self.judge_evaluator.evaluate_step(
                    task=self.memory.task,
                    thoughts=action_step.thoughts or "",
                    code=code,
                    result=action_step.observation or action_step.error or "",
                    step_number=step_number
                )
                
                action_step.evaluation = evaluation
                
                # Show evaluation in tracer
                self.tracer.add_evaluation(evaluation.dict())
                
                # Add the step to memory
                self.memory.add_step(action_step)
                
                # Optimize prompt if needed
                if self.optimize_prompt and self.memory.evaluations:
                    self.memory.system_prompt = self.judge_evaluator.optimize_prompt(
                        self.memory.system_prompt, 
                        self.memory.evaluations
                    )
                
                # Return the step output and not final
                return action_step.observation or action_step.error or "", False, token_stats
            else:
                # No code found, treat as a regular response
                action_step.thoughts = response
                self.memory.add_step(action_step)
                return response, True, token_stats
                
        else:
            # Tool use or regular chat mode
            action_step, is_final_answer = self._parse_response(response, step_number)
            
            # Store raw output and thoughts
            action_step.raw_output = response
            if thoughts and not action_step.thoughts:
                action_step.thoughts = thoughts
            
            # If this is a final answer, return it
            if is_final_answer:
                return action_step.thoughts or response, True, token_stats
                
            # If there's a tool call, execute it
            if action_step.action:
                try:
                    tool_name = action_step.action.tool_name
                    tool_input = action_step.action.tool_input
                    
                    # Find the matching tool
                    selected_tool = None
                    for tool in self.tools:
                        if tool.metadata.name == tool_name:
                            selected_tool = tool
                            break
                    
                    if not selected_tool:
                        error_msg = f"Tool '{tool_name}' not found"
                        action_step.error = error_msg
                        self.tracer.add_error(error_msg)
                    else:
                        # Check if this is a final_answer tool
                        if tool_name == "final_answer":
                            # Handle the final answer
                            answer = tool_input.get("answer", "")
                            action_step.observation = answer
                            self.memory.add_step(action_step)
                            return answer, True, token_stats
                        
                        # Execute the tool
                        self.tracer.add_tool_call(tool_name, tool_input)
                        tool_result = selected_tool(**tool_input)
                        
                        if tool_result.error:
                            action_step.error = tool_result.error
                            self.tracer.add_error(tool_result.error)
                        else:
                            # Format tool output as string if needed
                            observation = tool_result.content
                            if not isinstance(observation, str):
                                if hasattr(observation, "json"):
                                    # Pydantic model
                                    observation = observation.json()
                                else:
                                    # Convert to JSON string
                                    observation = json.dumps(observation, indent=2)
                            
                            action_step.observation = observation
                            self.tracer.add_output(observation)
                    
                except Exception as e:
                    action_step.error = f"Error executing tool: {str(e)}"
                    self.tracer.add_error(action_step.error)
                
                # Add the step to memory
                self.memory.add_step(action_step)
                
                # Return the step output and not final
                return action_step.observation or action_step.error or "", False, token_stats
            
            # No tool call found, treat as a regular response
            action_step.thoughts = response
            self.memory.add_step(action_step)
            return response, True, token_stats
    
    async def _generate_reasoning(self, task: str) -> ReasoningChain:
        """
        Generate reasoning for a task.
        
        Args:
            task: The task to reason about
            
        Returns:
            ReasoningChain with the reasoning steps
        """
        context = "Solve the task: " + task
        try:
            # Use direct reasoning through the adapter with no DSPy
            reasoning = self.reasoning_adapter.generate_reasoning(context, task)
            return reasoning
        except Exception as e:
            import traceback
            print(f"Error generating reasoning: {str(e)}")
            print(traceback.format_exc())
            # Return basic reasoning with the error
            return ReasoningChain(
                context=context,
                question=task,
                steps=[
                    ReasoningStep(
                        thoughts=f"Error generating reasoning: {str(e)}",
                        action_plan=["Proceed with task using available information"]
                    )
                ]
            )
    
    async def _generate_final_answer(self) -> str:
        """
        Generate a final answer after running out of steps.
        
        Returns:
            Final answer string
        """
        messages = self._create_messages()
        
        # Add a special message prompting for a final answer
        messages.append({
            "role": "user",
            "content": "Please provide your final answer based on all steps so far. Summarize what you've learned and give a complete response."
        })
        
        # Call the model
        response = await self._call_model(messages)
        
        return response
    
    async def _call_model(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the model with messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Model response as string
        """
        if hasattr(self.client, 'chat') and inspect.iscoroutinefunction(self.client.chat):
            return await self.client.chat(messages)
        else:
            # Fallback to synchronous call using anyio.to_thread.run_sync
            return await anyio.to_thread.run_sync(lambda: self.client(messages))
    
    def _create_messages(self) -> List[Dict[str, str]]:
        """
        Create messages for the model based on the current memory.
        
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self._create_system_message()
        })
        
        # Add user task message
        messages.append({
            "role": "user", 
            "content": self.memory.task
        })
        
        # Add previous steps as alternating user/assistant messages
        for i, step in enumerate(self.memory.steps):
            # Add assistant's action/code
            assistant_content = ""
            
            if step.thoughts:
                assistant_content += f"{step.thoughts}\n\n"
            
            if step.action:
                assistant_content += f"I'll use the {step.action.tool_name} tool with these parameters: {json.dumps(step.action.tool_input)}"
            
            if step.code:
                assistant_content += f"```python\n{step.code}\n```"
                
            if assistant_content:
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            # Add observation/error as user message
            user_content = ""
            if step.observation:
                user_content += f"Observation: {step.observation}"
            if step.error:
                user_content += f"Error: {step.error}"
                
            if user_content:
                messages.append({
                    "role": "user",
                    "content": user_content
                })
                
            # Add judge evaluation feedback as user message if available
            if step.evaluation and step.evaluation.feedback:
                messages.append({
                    "role": "user",
                    "content": f"Feedback on your previous step: {step.evaluation.feedback} (Score: {step.evaluation.evaluation_score:.1f}/1.0)"
                })
                
        return messages
    
    def _create_system_message(self) -> str:
        """
        Create the system message for the agent.
        
        Returns:
            System message string
        """
        system_message = self.memory.system_prompt
        
        # Add appropriate instructions based on mode
        if self.mode == AgentMode.TOOL_USE and self.tools:
            system_message += self._create_tool_instructions()
        elif self.mode == AgentMode.CODE_EXECUTION:
            system_message += self._create_code_execution_instructions()
            
        # Add final_answer tool instructions if available
        if any(tool.metadata.name == "final_answer" for tool in self.tools):
            system_message += """

IMPORTANT: When you have completed the task and are ready to provide a final answer, 
you MUST use the final_answer tool to deliver your response. The conversation will 
continue until you use this tool. Use the following format:

```json
{
  "name": "final_answer",
  "arguments": {
    "answer": "Your complete and final answer here"
  }
}
```

Alternatively, you can use the function-style call:
final_answer("Your complete and final answer here")

Do not provide a final answer without using this tool.
"""
            
        return system_message
    
    def _create_tool_instructions(self) -> str:
        """Create instructions for tool use mode."""
        tool_descriptions = "\n\nYou have access to the following tools:\n\n"
        
        for tool in self.tools:
            meta = tool.metadata
            tool_descriptions += f"Tool: {meta.name}\n"
            tool_descriptions += f"Description: {meta.description}\n"
            
            if meta.parameters:
                tool_descriptions += "Parameters:\n"
                for param in meta.parameters:
                    required_str = "required" if param.required else "optional"
                    tool_descriptions += f"- {param.name} ({param.type}, {required_str}): {param.description}\n"
                
            tool_descriptions += "\n"
            
        # Add instructions for tool use
        tool_descriptions += """
When you need to use a tool, respond using the following format:

I'll analyze this step by step.
[Your reasoning here]

I'll use the [tool name] tool with these parameters: {"param1": "value1", "param2": "value2", ...}

When you have completed the task and are confident in your final answer, you MUST use the final_answer tool:

```json
{
  "name": "final_answer",
  "arguments": {
    "answer": "Your complete and final answer here"
  }
}
```

Never provide a final answer without using the final_answer tool. The conversation will continue until you use this tool.
"""
        return tool_descriptions
    
    def _create_code_execution_instructions(self) -> str:
        """Create instructions for code execution mode."""
        code_instructions = """

You are a coding assistant that writes and executes Python code to solve problems.

When solving a problem, write out your reasoning and then provide executable Python code.
Your code will be run in a Python environment with the following limitations:
- Limited stdlib modules are available
- No access to the internet
- Limited execution time

Format your response like this:
1. Write your reasoning about how to solve the problem
2. Provide a complete Python solution within a single code block:
```python
# Your code here
```

After your code runs, you'll see the output and can refine your solution if needed.
Always ensure your code is complete, properly handles edge cases, and follows best practices.

Once you are confident that your code is correct and the task is complete, do NOT just provide a text summary.
Instead, you MUST use the final_answer tool to provide your final result. For example:

```json
{
  "name": "final_answer",
  "arguments": {
    "answer": "The solution is [result]. The code successfully [description of what the code accomplishes]."
  }
}
```

Or you can use the function format: final_answer("Your complete answer here")

Your goal is to solve the problem correctly and efficiently through step-by-step code execution,
and then provide the final answer using the final_answer tool.
"""
        return code_instructions
    
    def _parse_response(
        self, 
        response: str, 
        step_number: int
    ) -> Tuple[ActionStep, bool]:
        """
        Parse a response from the model into an ActionStep.
        
        Args:
            response: Response from the model
            step_number: Step number
            
        Returns:
            Tuple of (ActionStep, is_final_answer)
        """
        action_step = ActionStep(step_number=step_number)
        is_final_answer = False
        
        # Check for final answer in the text response
        final_answer_pattern = r"final_answer\((.*?)\)"
        final_answer_match = re.search(final_answer_pattern, response, re.DOTALL)
        
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            # Remove quotes if present
            if (answer.startswith('"') and answer.endswith('"')) or \
               (answer.startswith("'") and answer.endswith("'")):
                answer = answer[1:-1]
                
            # Create a final answer tool call
            action_step.action = ToolCall(
                tool_name="final_answer",
                tool_input={"answer": answer}
            )
            
            # Extract thoughts before the final answer
            thoughts_text = response[:final_answer_match.start()].strip()
            if thoughts_text:
                action_step.thoughts = thoughts_text
                
            return action_step, True
        
        # Check for tool calls in JSON format
        tool_pattern = r"```(?:json)?\s*\{[\s\S]*?\"(?:action|name|tool)\"[\s\S]*?\}(?:\s*\n)?\s*```"
        tool_matches = re.finditer(tool_pattern, response)
        
        for match in tool_matches:
            tool_json_str = match.group(0).strip()
            # Remove markdown code block syntax
            tool_json_str = re.sub(r"```(?:json)?\s*", "", tool_json_str)
            tool_json_str = re.sub(r"\s*```", "", tool_json_str)
            
            try:
                tool_data = json.loads(tool_json_str)
                
                # Handle different formats (openai vs custom)
                if "action" in tool_data:
                    tool_name = tool_data["action"]
                    tool_input = tool_data.get("action_input", {})
                elif "name" in tool_data:
                    tool_name = tool_data["name"]
                    tool_input = tool_data.get("arguments", {})
                    # Parse string arguments as JSON if possible
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except:
                            tool_input = {"input": tool_input}
                elif "tool" in tool_data:
                    tool_name = tool_data["tool"]
                    tool_input = tool_data.get("tool_input", {})
                else:
                    continue
                
                # Special handling for final_answer tool
                if tool_name == "final_answer":
                    is_final_answer = True
                    
                    # Extract answer from various possible formats
                    if isinstance(tool_input, dict) and "answer" in tool_input:
                        answer = tool_input["answer"]
                    elif isinstance(tool_input, str):
                        answer = tool_input
                    else:
                        answer = str(tool_input)
                        
                    tool_input = {"answer": answer}
                
                action_step.action = ToolCall(
                    tool_name=tool_name,
                    tool_input=tool_input
                )
                
                # Extract thoughts before the tool call
                thoughts_text = response[:match.start()].strip()
                if thoughts_text:
                    action_step.thoughts = thoughts_text
                    
                break
            
            except Exception as e:
                continue
                
        if not action_step.action and not is_final_answer:
            # No tool call found, the entire response is thoughts
            action_step.thoughts = response
            is_final_answer = True
            
        return action_step, is_final_answer
    
    def _parse_code_response(
        self,
        response: str,
        step_number: int
    ) -> Tuple[ActionStep, bool, Optional[str]]:
        """
        Parse a code response from the model.
        
        Args:
            response: Model response string
            step_number: Current step number
            
        Returns:
            Tuple of (ActionStep, is_final_answer, code)
        """
        # Create a basic action step
        action_step = ActionStep(step_number=step_number)
        
        # First check for final_answer tool calls using the function format
        final_answer_pattern = r"final_answer\((.*?)\)"
        final_answer_match = re.search(final_answer_pattern, response, re.DOTALL)
        
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            # Remove quotes if present
            if (answer.startswith('"') and answer.endswith('"')) or \
               (answer.startswith("'") and answer.endswith("'")):
                answer = answer[1:-1]
                
            # Create a final answer tool call
            action_step.action = ToolCall(
                tool_name="final_answer",
                tool_input={"answer": answer}
            )
            
            # Extract thoughts before the final answer
            thoughts_text = response[:final_answer_match.start()].strip()
            if thoughts_text:
                action_step.thoughts = thoughts_text
                
            return action_step, True, None
        
        # Check for JSON format tool calls, especially final_answer
        tool_pattern = r"```(?:json)?\s*\{[\s\S]*?\"(?:action|name|tool)\"[\s\S]*?\}(?:\s*\n)?\s*```"
        tool_matches = re.finditer(tool_pattern, response)
        
        for match in tool_matches:
            tool_json_str = match.group(0).strip()
            # Remove markdown code block syntax
            tool_json_str = re.sub(r"```(?:json)?\s*", "", tool_json_str)
            tool_json_str = re.sub(r"\s*```", "", tool_json_str)
            
            try:
                tool_data = json.loads(tool_json_str)
                
                # Handle different formats (openai vs custom)
                if "action" in tool_data:
                    tool_name = tool_data["action"]
                    tool_input = tool_data.get("action_input", {})
                elif "name" in tool_data:
                    tool_name = tool_data["name"]
                    tool_input = tool_data.get("arguments", {})
                    # Parse string arguments as JSON if possible
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except:
                            tool_input = {"input": tool_input}
                elif "tool" in tool_data:
                    tool_name = tool_data["tool"]
                    tool_input = tool_data.get("tool_input", {})
                else:
                    continue
                
                # Handle final_answer tool specifically
                if tool_name == "final_answer":
                    # Extract answer from various possible formats
                    if isinstance(tool_input, dict) and "answer" in tool_input:
                        answer = tool_input["answer"]
                    elif isinstance(tool_input, str):
                        answer = tool_input
                    else:
                        answer = str(tool_input)
                        
                    action_step.action = ToolCall(
                        tool_name="final_answer",
                        tool_input={"answer": answer}
                    )
                    
                    # Extract thoughts before the tool call
                    thoughts_text = response[:match.start()].strip()
                    if thoughts_text:
                        action_step.thoughts = thoughts_text
                        
                    return action_step, True, None
            except:
                # Failed JSON parsing, continue with code block check
                pass
        
        # Next, check for code blocks
        code_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        code_match = re.search(code_pattern, response, re.DOTALL)
        
        if code_match:
            # Extract code and thoughts
            code = code_match.group(1).strip()
            
            # Extract thoughts (everything before code block)
            thoughts_parts = response.split("```", 1)
            thoughts = thoughts_parts[0].strip()
            
            action_step.thoughts = thoughts
            
            return action_step, False, code
        else:
            # No code found, treat as final answer
            action_step.thoughts = response
            return action_step, True, None 
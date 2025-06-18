"""
Enhanced DSPy integration for robust reasoning capabilities.
"""

import re
import json
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pydantic import BaseModel, Field

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("Warning: DSPy not available. Advanced reasoning capabilities will be disabled.")

from .client.huggingface_client import HuggingFaceClient, Message

class ReasoningStep(BaseModel):
    """A step in the reasoning process."""
    thoughts: str = Field(..., description="Agent's internal thoughts for this step")
    action_plan: List[str] = Field(default_factory=list, description="Planned actions for this step")
    
class ReasoningChain(BaseModel):
    """A chain of reasoning steps."""
    context: str = Field(..., description="Context for the reasoning chain")
    question: str = Field(..., description="Question or task to reason about")
    steps: List[ReasoningStep] = Field(default_factory=list, description="Steps in the reasoning chain")
    conclusion: Optional[str] = Field(None, description="Final conclusion")

class DSPyReasoningAdapter:
    """
    Enhanced DSPy-based reasoning adapter that works with or without DSPy.
    Provides fallback to direct prompting when DSPy is not available.
    """
    
    def __init__(
        self,
        model: Union[HuggingFaceClient, Callable],
        use_dspy: bool = True,
        max_iterations: int = 3,
        temperature: float = 0.7,
    ):
        """
        Initialize the DSPy reasoning adapter.
        
        Args:
            model: The model to use for reasoning
            use_dspy: Whether to use DSPy for reasoning (if available)
            max_iterations: Maximum number of reasoning iterations
            temperature: Temperature for reasoning generation
        """
        self.model = model
        self.use_dspy = use_dspy and DSPY_AVAILABLE
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Set up DSPy if available
        if self.use_dspy:
            try:
                self._setup_dspy()
                print("DSPy reasoning setup successful")
            except Exception as e:
                print(f"Error setting up DSPy: {str(e)}")
                self.use_dspy = False
                self.pot_module = None
        else:
            self.pot_module = None
            
    def _setup_dspy(self):
        """Set up DSPy for reasoning."""
        if not DSPY_AVAILABLE:
            return
            
        # Create a DSPy LM wrapper
        self.dspy_lm = self._create_dspy_lm_wrapper()
        
        # Configure DSPy to use our LM
        dspy.configure(lm=self.dspy_lm)
        
        # Define the reasoning signature
        class ReasoningSignature(dspy.Signature):
            """Signature for step-by-step reasoning with DSPy."""
            context = dspy.InputField(desc="The context including all relevant information")
            question = dspy.InputField(desc="The question or task to solve")
            reasoning = dspy.OutputField(desc="Step-by-step reasoning to solve the task")
            next_steps = dspy.OutputField(desc="Concrete next steps to take to make progress")
        
        # Create the ProgramOfThought module
        self.pot_module = dspy.ProgramOfThought(
            ReasoningSignature,
            max_iters=self.max_iterations
        )
    
    def _create_dspy_lm_wrapper(self):
        """Create a DSPy LM wrapper for the model."""
        class CustomDSPyLM(dspy.LM):
            """Custom DSPy LM wrapper for our model."""
            
            def __init__(self, model):
                # Initialize with custom provider and model name
                super().__init__(model="custom/reasoning-model", provider="custom")
                self.model = model
                # Add model_type for DSPy compatibility
                self.model_type = getattr(model, 'model_type', 'chat')
                
            def basic_request(self, prompt, **kwargs):
                """Basic request implementation for DSPy."""
                try:
                    # Format as a message
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Extract temperature parameter
                    temp = kwargs.get('temperature', 0.7)
                    
                    # Call the model
                    if hasattr(self.model, '__call__'):
                        # Use synchronous call
                        return self.model(messages, temperature=temp)
                    else:
                        # Shouldn't reach here as we handle async calls in completion method
                        return "Model call method not available"
                        
                except Exception as e:
                    print(f"Warning: Error in DSPy LM request: {str(e)}")
                    return "Failed to generate reasoning."
            
            # Override completion for DSPy
            def completion(self, prompt, **kwargs):
                response_text = self.basic_request(prompt, **kwargs)
                return {"choices": [{"text": response_text}]}
                
            # Provide required DSPy compatibility methods
            def get_llm_model_name(self):
                return "custom-reasoning-model"
                
        return CustomDSPyLM(self.model)
        
    def generate_reasoning(
        self, 
        context: str, 
        question: str
    ) -> ReasoningChain:
        """
        Generate a chain of reasoning for the given context and question.
        
        Args:
            context: Context information
            question: Question or task to reason about
            
        Returns:
            ReasoningChain with the reasoning steps
        """
        reasoning_chain = ReasoningChain(
            context=context,
            question=question,
            steps=[]
        )
        
        if self.use_dspy and self.pot_module:
            # Use DSPy for reasoning
            try:
                print("Using DSPy for reasoning generation")
                pot_result = self.pot_module(context=context, question=question)
                
                # Parse the DSPy result
                reasoning_text = pot_result.reasoning if hasattr(pot_result, 'reasoning') else ""
                next_steps = pot_result.next_steps if hasattr(pot_result, 'next_steps') else ""
                
                # Split reasoning into steps
                steps = self._parse_reasoning_steps(reasoning_text)
                
                for i, step in enumerate(steps):
                    reasoning_chain.steps.append(
                        ReasoningStep(
                            thoughts=step,
                            action_plan=[] if i < len(steps) - 1 else self._parse_action_items(next_steps)
                        )
                    )
                
                reasoning_chain.conclusion = next_steps
                return reasoning_chain
                
            except Exception as e:
                import traceback
                print(f"Warning: DSPy reasoning failed, falling back to direct prompting: {str(e)}")
                print(traceback.format_exc())
                # Fall back to direct prompting
        
        # Direct prompting fallback
        print("Using direct prompting for reasoning generation")
        reasoning_prompt = self._create_reasoning_prompt(context, question)
        
        # Send to model
        messages = [{"role": "user", "content": reasoning_prompt}]
        
        try:
            # Handle different model interfaces
            if hasattr(self.model, '__call__'):
                # Synchronous call
                response = self.model(messages, temperature=self.temperature)
            elif hasattr(self.model, 'chat') and inspect.iscoroutinefunction(self.model.chat):
                # Use anyio to run async function
                import anyio
                response = anyio.run(
                    self.model.chat, 
                    messages, 
                    temperature=self.temperature
                )
            else:
                # No compatible interface
                raise ValueError("Model does not have a compatible interface")
                
            # Parse the response
            reasoning_chain = self._parse_reasoning_response(response, context, question)
            return reasoning_chain
            
        except Exception as e:
            import traceback
            print(f"Warning: Direct reasoning failed: {str(e)}")
            print(traceback.format_exc())
            # Return basic reasoning with the error
            return ReasoningChain(
                context=context,
                question=question,
                steps=[
                    ReasoningStep(
                        thoughts=f"Error generating reasoning: {str(e)}",
                        action_plan=["Proceed with task using available information"]
                    )
                ]
            )
    
    def _create_reasoning_prompt(self, context: str, question: str) -> str:
        """Create a prompt for direct reasoning generation."""
        return f"""Please analyze the following task and provide step-by-step reasoning.

CONTEXT:
{context}

TASK:
{question}

Think about this step-by-step. First identify what you know, then analyze what you need to determine, 
and finally work through the solution logically. 

Your response should be structured like this:
REASONING:
- Step 1: [Your first step of reasoning]
- Step 2: [Your second step of reasoning]
...
- Step N: [Your final step of reasoning and conclusion]

NEXT STEPS:
1. [First concrete action to take]
2. [Second concrete action to take]
...

Begin your analysis now:
"""

    def _parse_reasoning_response(
        self, 
        response: str, 
        context: str, 
        question: str
    ) -> ReasoningChain:
        """Parse a direct reasoning response into a ReasoningChain."""
        chain = ReasoningChain(
            context=context,
            question=question,
            steps=[]
        )
        
        # Extract reasoning steps
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?:NEXT STEPS:|$)", response, re.DOTALL)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            steps = self._parse_reasoning_steps(reasoning_text)
            
            # Extract next steps
            next_steps_match = re.search(r"NEXT STEPS:\s*(.*?)$", response, re.DOTALL)
            next_steps = next_steps_match.group(1).strip() if next_steps_match else ""
            action_items = self._parse_action_items(next_steps)
            
            # Create steps
            for i, step in enumerate(steps):
                chain.steps.append(
                    ReasoningStep(
                        thoughts=step,
                        action_plan=[] if i < len(steps) - 1 else action_items
                    )
                )
                
            chain.conclusion = next_steps
        else:
            # Couldn't parse structured format, treat whole response as reasoning
            chain.steps.append(
                ReasoningStep(
                    thoughts=response,
                    action_plan=[]
                )
            )
            
        return chain
    
    def _parse_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """Parse reasoning text into steps."""
        # Try to extract steps using numbered or bulleted format
        step_pattern = r"(?:Step \d+:|[-•*]\s)(.*?)(?=Step \d+:|[-•*]\s|$)"
        steps = re.findall(step_pattern, reasoning_text, re.DOTALL)
        
        if not steps:
            # If structured steps not found, split by newlines as fallback
            steps = [line.strip() for line in reasoning_text.split('\n') if line.strip()]
            
        if not steps:
            # Last resort: use the whole text as one step
            steps = [reasoning_text.strip()]
            
        return [step.strip() for step in steps]
    
    def _parse_action_items(self, next_steps: str) -> List[str]:
        """Parse next steps text into action items."""
        # Look for numbered or bulleted format
        action_pattern = r"(?:\d+\.|\*|\-)\s*(.*?)(?=(?:\d+\.|\*|\-)|$)"
        actions = re.findall(action_pattern, next_steps, re.DOTALL)
        
        if not actions:
            # If structured actions not found, split by newlines as fallback
            actions = [line.strip() for line in next_steps.split('\n') if line.strip()]
            
        return [action.strip() for action in actions]
    
    def add_reasoning_to_system_prompt(
        self, 
        system_prompt: str, 
        reasoning: ReasoningChain
    ) -> str:
        """
        Add reasoning to a system prompt.
        
        Args:
            system_prompt: Original system prompt
            reasoning: ReasoningChain to add
            
        Returns:
            Enhanced system prompt with reasoning
        """
        reasoning_text = "REASONING:\n"
        
        for i, step in enumerate(reasoning.steps):
            reasoning_text += f"Step {i+1}: {step.thoughts}\n"
            
        if reasoning.conclusion:
            reasoning_text += f"\nCONCLUSION:\n{reasoning.conclusion}"
            
        # Add the reasoning to the system prompt
        if "Here's some useful reasoning:" in system_prompt:
            # Replace existing reasoning
            parts = system_prompt.split("Here's some useful reasoning:")
            system_prompt = parts[0] + f"Here's some useful reasoning:\n{reasoning_text}"
        else:
            # Add new reasoning
            system_prompt += f"\n\nHere's some useful reasoning:\n{reasoning_text}"
            
        return system_prompt 
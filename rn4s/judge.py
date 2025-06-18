"""
Judge evaluation system for agent steps.
"""

import re
import json
import inspect
from typing import Any, Dict, List, Optional, Union, Callable
import anyio
from pydantic import BaseModel, Field

from .client.huggingface_client import HuggingFaceClient

class JudgeConfig(BaseModel):
    """Configuration for the judge evaluator."""
    prompt_template: str = Field(..., description="Template for the judge prompt")
    criteria: List[str] = Field(default_factory=list, description="Criteria for evaluation")
    evaluation_threshold: float = Field(0.7, description="Threshold for good evaluation")
    weight_quality: float = Field(0.6, description="Weight for quality assessment")
    weight_relevance: float = Field(0.4, description="Weight for relevance assessment")

class JudgeEvaluation(BaseModel):
    """Structured evaluation from the judge."""
    step_number: int = Field(..., description="Step number in the agent execution")
    evaluation_score: float = Field(..., description="Evaluation score from 0.0 to 1.0")
    feedback: str = Field(..., description="Feedback from the judge about the quality of the response")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the evaluation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class JudgeEvaluator:
    """
    Evaluator that judges agent steps and provides structured feedback.
    """
    
    def __init__(
        self,
        client: Union[HuggingFaceClient, Callable],
        config: Optional[JudgeConfig] = None,
    ):
        """
        Initialize the judge evaluator.
        
        Args:
            client: The model client to use for evaluations
            config: Configuration for the evaluator
        """
        self.client = client
        self.config = config or self._create_default_config()
        
    def _create_default_config(self) -> JudgeConfig:
        """Create default judge configuration."""
        return JudgeConfig(
            prompt_template=DEFAULT_JUDGE_PROMPT,
            criteria=DEFAULT_JUDGE_CRITERIA
        )
    
    async def evaluate_step(
        self, 
        task: str,
        thoughts: str,
        code: Optional[str],
        result: str,
        step_number: int
    ) -> JudgeEvaluation:
        """
        Evaluate an agent step.
        
        Args:
            task: The original task/question
            thoughts: The agent's reasoning/thoughts
            code: The code executed (if any)
            result: The result or observation
            step_number: The step number
            
        Returns:
            JudgeEvaluation with structured feedback
        """
        # Construct the judge prompt
        prompt = self._construct_judge_prompt(task, thoughts, code, result)
        
        # Send to model for evaluation
        messages = [{"role": "user", "content": prompt}]
        
        # Call the model
        if hasattr(self.client, 'chat') and inspect.iscoroutinefunction(self.client.chat):
            response = await self.client.chat(messages, temperature=0.2)
        else:
            # Use anyio to call synchronous function
            response = await anyio.to_thread.run_sync(
                lambda: self.client(messages)
            )
        
        try:
            # Parse the judge response
            evaluation = self._parse_judge_response(response, step_number)
            return evaluation
        except Exception as e:
            # If parsing fails, return a default evaluation
            return JudgeEvaluation(
                step_number=step_number,
                evaluation_score=0.5,
                feedback=f"Failed to parse judge response: {str(e)}",
                reasoning="Evaluation parsing error",
                metadata={"raw_response": response}
            )
    
    def _construct_judge_prompt(
        self, 
        task: str, 
        thoughts: str, 
        code: Optional[str], 
        result: str
    ) -> str:
        """Construct the prompt for the judge model."""
        prompt = self.config.prompt_template
        
        # Replace placeholders in the template
        prompt = prompt.replace("{QUESTION}", task)
        prompt = prompt.replace("{REASONING}", thoughts or "No reasoning provided")
        
        # For the answer, include both code and result if code exists
        answer = ""
        if code:
            answer = f"Code:\n{code}\n\nResult:\n{result}"
        else:
            answer = result
            
        prompt = prompt.replace("{ANSWER}", answer)
        prompt = prompt.replace("{CRITERIA}", "\n".join([f"- {c}" for c in self.config.criteria]))
        
        return prompt
    
    def _parse_judge_response(self, response: str, step_number: int) -> JudgeEvaluation:
        """Parse the judge response into a JudgeEvaluation object."""
        # Extract score using regex
        score_match = re.search(r"Score:\s*(\d+\.?\d*)", response)
        score = float(score_match.group(1)) if score_match else 0.5
        
        # Extract feedback using regex
        feedback_match = re.search(r"Feedback:(.*?)(?:Reasoning:|$)", response, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided"
        
        # Extract reasoning using regex
        reasoning_match = re.search(r"Reasoning:(.*)", response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        
        return JudgeEvaluation(
            step_number=step_number,
            evaluation_score=score,
            feedback=feedback,
            reasoning=reasoning
        )
    
    def optimize_prompt(self, system_prompt: str, evaluations: List[JudgeEvaluation]) -> str:
        """
        Optimize a system prompt based on judge evaluations.
        
        Args:
            system_prompt: Original system prompt
            evaluations: List of evaluations from judge
            
        Returns:
            Optimized system prompt
        """
        if not evaluations:
            return system_prompt
            
        # Create a summary of past feedback, prioritizing recent and higher-scored evaluations
        sorted_evals = sorted(
            evaluations[-5:],  # Consider recent evaluations 
            key=lambda x: x.evaluation_score, 
            reverse=True
        )
        
        # Format the feedback summary
        feedback_summary = "\n\n".join([
            f"Feedback from step {eval.step_number} (score: {eval.evaluation_score:.1f}): {eval.feedback}"
            for eval in sorted_evals[:3]  # Use top 3 evaluations
        ])
        
        # Add the feedback to the system prompt
        if "PREVIOUS_FEEDBACK" in system_prompt:
            # Replace placeholder
            system_prompt = system_prompt.replace("PREVIOUS_FEEDBACK", feedback_summary)
        elif "Consider this feedback from previous steps:" in system_prompt:
            # Replace existing feedback
            parts = system_prompt.split("Consider this feedback from previous steps:")
            system_prompt = parts[0] + f"Consider this feedback from previous steps:\n{feedback_summary}"
        else:
            # Add new feedback section
            system_prompt += f"\n\nConsider this feedback from previous steps:\n{feedback_summary}"
            
        return system_prompt


# Default judge prompt template
DEFAULT_JUDGE_PROMPT = """
You are an expert judge evaluating the quality of an AI assistant's reasoning, code and answer.

Question: {QUESTION}

AI Reasoning: {REASONING}

AI Answer: {ANSWER}

Your task is to evaluate the quality of the AI's reasoning and answer based on the following criteria:
{CRITERIA}

Instructions:
1. Analyze the AI's reasoning, code (if any), and answer quality
2. Provide a score from 0.0 to 1.0, where:
   - 0.0: Completely incorrect or harmful
   - 0.5: Partially correct with significant issues
   - 1.0: Excellent, accurate, and helpful

3. Give specific feedback on the strengths and weaknesses
4. Suggest improvements for future steps

Respond in the following format:
Score: [your score between 0.0 and 1.0]
Feedback: [your detailed feedback]
Reasoning: [explain your evaluation]
"""

# Default evaluation criteria
DEFAULT_JUDGE_CRITERIA = [
    "Correctness: Is the reasoning and code factually accurate?",
    "Relevance: Does the answer directly address the question?",
    "Completeness: Does the answer cover all aspects of the question?",
    "Code Quality: Is the code well-structured and efficient? (if applicable)",
    "Logic: Is the reasoning process logical and sound?",
    "Execution: Did the code execute correctly and produce the expected result? (if applicable)"
] 
"""
Formatting utilities for agent execution traces, inspired by smolagents.
Provides rich console output for agent execution steps and results.
"""

import time
import sys
import json
import textwrap
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import re
from datetime import datetime

# Try importing rich for better formatting if available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ANSI color codes for basic terminal formatting when rich is not available
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

class StepType(str, Enum):
    """Types of steps that can be displayed."""
    TASK = "task"
    STEP = "step"
    CODE = "code"
    OUTPUT = "output"
    ERROR = "error"
    EVALUATION = "evaluation"
    FINAL = "final"
    THINKING = "thinking"
    
class AgentTracer:
    """
    Tracer for agent execution that formats and displays agent steps in a smolagents style.
    """
    
    def __init__(self, verbose: bool = True, show_thinking: bool = False):
        """
        Initialize the tracer.
        
        Args:
            verbose: Whether to print detailed output
            show_thinking: Whether to show thinking steps (which can be very verbose)
        """
        self.verbose = verbose
        self.show_thinking = show_thinking
        self.console = Console() if RICH_AVAILABLE else None
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = 0
        self.token_counts = {}
        
    def start_task(self, task: str):
        """
        Start a new task.
        
        Args:
            task: The task description
        """
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = 0
        self.token_counts = {}
        
        self._print_section(StepType.TASK, task)
    
    def start_step(self, step_number: int):
        """
        Start a new step.
        
        Args:
            step_number: The step number
        """
        self.current_step = step_number
        self.step_times[step_number] = time.time()
        
        self._print_section(StepType.STEP, f"Step {step_number}")
    
    def add_code(self, code: str):
        """
        Add a code snippet.
        
        Args:
            code: The code snippet
        """
        self._print_section(StepType.CODE, code)
    
    def add_output(self, output: str):
        """
        Add output from code execution.
        
        Args:
            output: The execution output
        """
        self._print_section(StepType.OUTPUT, output)
    
    def add_error(self, error: str):
        """
        Add an error message.
        
        Args:
            error: The error message
        """
        self._print_section(StepType.ERROR, error)
    
    def add_thinking(self, thinking: str):
        """
        Add thinking/reasoning.
        
        Args:
            thinking: The thinking text
        """
        if self.show_thinking:
            self._print_section(StepType.THINKING, thinking)
    
    def add_evaluation(self, evaluation: Dict[str, Any]):
        """
        Add evaluation results.
        
        Args:
            evaluation: The evaluation details
        """
        eval_text = f"Score: {evaluation.get('evaluation_score', 0):.2f}/1.0\n"
        eval_text += f"Feedback: {evaluation.get('feedback', '')}"
        
        self._print_section(StepType.EVALUATION, eval_text)
    
    def end_step(self, token_stats: Optional[Dict[str, int]] = None):
        """
        End the current step and print summary.
        
        Args:
            token_stats: Optional dictionary with token counts
        """
        if self.current_step in self.step_times:
            elapsed = time.time() - self.step_times[self.current_step]
            
            summary = [f"- Time taken: {elapsed:.2f} seconds"]
            
            if token_stats:
                self.token_counts[self.current_step] = token_stats
                summary.append(f"- Input tokens: {token_stats.get('input_tokens', 0)}")
                summary.append(f"- Output tokens: {token_stats.get('output_tokens', 0)}")
            
            self._print_plain("\n" + "\n".join(summary))
    
    def final_answer(self, answer: str):
        """
        Print the final answer.
        
        Args:
            answer: The final answer
        """
        self._print_section(StepType.FINAL, answer)
        
        # Print total execution time
        total_time = time.time() - self.start_time
        total_input_tokens = sum(stats.get('input_tokens', 0) for stats in self.token_counts.values())
        total_output_tokens = sum(stats.get('output_tokens', 0) for stats in self.token_counts.values())
        
        summary = [
            f"Total execution time: {total_time:.2f} seconds",
            f"Total input tokens: {total_input_tokens}",
            f"Total output tokens: {total_output_tokens}"
        ]
        
        self._print_plain("\n" + "\n".join(summary))
    
    def _print_section(self, section_type: StepType, content: str):
        """
        Print a formatted section.
        
        Args:
            section_type: The type of section
            content: The content to print
        """
        if not self.verbose:
            return
            
        if RICH_AVAILABLE:
            self._print_rich_section(section_type, content)
        else:
            self._print_plain_section(section_type, content)
    
    def _print_rich_section(self, section_type: StepType, content: str):
        """
        Print a section using rich formatting.
        
        Args:
            section_type: The type of section
            content: The content to print
        """
        if section_type == StepType.TASK:
            title = "New task"
            style = "bold green"
            self.console.print(Panel(content, title=title, expand=False, border_style=style))
        
        elif section_type == StepType.STEP:
            self.console.print(f"\n[bold blue]{content}[/bold blue]")
            self.console.print("─" * min(100, self.console.width))
        
        elif section_type == StepType.CODE:
            self.console.print("[bold cyan]Agent is executing the code below:[/bold cyan]")
            syntax = Syntax(content, "python", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        
        elif section_type == StepType.OUTPUT:
            self.console.print("[bold green]Output from code execution:[/bold green]")
            if content.strip():
                self.console.print(Panel(content, expand=False))
            else:
                self.console.print("[italic](No output)[/italic]")
        
        elif section_type == StepType.ERROR:
            self.console.print("[bold red]Error:[/bold red]")
            self.console.print(Panel(content, border_style="red", expand=False))
        
        elif section_type == StepType.EVALUATION:
            self.console.print("[bold magenta]Evaluation:[/bold magenta]")
            self.console.print(Panel(content, border_style="magenta", expand=False))
        
        elif section_type == StepType.FINAL:
            self.console.print("\n[bold green]Final answer:[/bold green]")
            self.console.print(Panel(content, border_style="green", expand=False))
            
        elif section_type == StepType.THINKING:
            self.console.print("[bold yellow]Thinking:[/bold yellow]")
            self.console.print(Panel(content, border_style="yellow", expand=False))
    
    def _print_plain_section(self, section_type: StepType, content: str):
        """
        Print a section using plain ANSI formatting.
        
        Args:
            section_type: The type of section
            content: The content to print
        """
        if section_type == StepType.TASK:
            separator = "=" * 80
            print(f"\n{separator}")
            print(f"{Color.BOLD}{Color.GREEN}New task{Color.RESET}")
            print(separator)
            print(content)
            
        elif section_type == StepType.STEP:
            separator = "-" * 80
            print(f"\n{separator}")
            print(f"{Color.BOLD}{Color.BLUE}{content}{Color.RESET}")
            print(separator)
            
        elif section_type == StepType.CODE:
            print(f"{Color.BOLD}{Color.CYAN}Agent is executing the code below:{Color.RESET}")
            print(f"{Color.BG_BLACK}")
            # Add simple line numbers
            lines = content.split("\n")
            for i, line in enumerate(lines):
                print(f"{i+1:3d} | {line}")
            print(f"{Color.RESET}")
            
        elif section_type == StepType.OUTPUT:
            print(f"{Color.BOLD}{Color.GREEN}Output from code execution:{Color.RESET}")
            if content.strip():
                print(content)
            else:
                print(f"{Color.BOLD}(No output){Color.RESET}")
                
        elif section_type == StepType.ERROR:
            print(f"{Color.BOLD}{Color.RED}Error:{Color.RESET}")
            print(content)
            
        elif section_type == StepType.EVALUATION:
            print(f"{Color.BOLD}{Color.MAGENTA}Evaluation:{Color.RESET}")
            print(content)
            
        elif section_type == StepType.FINAL:
            print(f"\n{Color.BOLD}{Color.GREEN}Final answer:{Color.RESET}")
            print(content)
            
        elif section_type == StepType.THINKING:
            print(f"{Color.BOLD}{Color.YELLOW}Thinking:{Color.RESET}")
            print(content)
    
    def _print_plain(self, content: str):
        """
        Print plain text without special formatting.
        
        Args:
            content: The content to print
        """
        if not self.verbose:
            return
            
        if RICH_AVAILABLE:
            self.console.print(content)
        else:
            print(content)


class ThinkOutputParser:
    """
    Parser to extract thought/reasoning from model outputs, especially from models
    that add explicit <think> tags or similar thought markers.
    """
    
    def __init__(self):
        # Patterns for different thought formats
        self.patterns = [
            # QWQ and similar models using <think> tags
            (r'<think>(.*?)</think>', 1),
            # Verbalized thinking with prefixes like "I'll think through this step by step"
            (r"(?:I'll think through this step by step|Let me think about this)(.*?)(?=\n\n|$)", 1),
            # Thought: prefix used by some models
            (r"Thought:(.*?)(?=\n\n|Action:|Observation:|$)", 1)
        ]
    
    def extract_thoughts(self, text: str) -> Optional[str]:
        """
        Extract thinking/reasoning from text.
        
        Args:
            text: The text to parse
            
        Returns:
            Extracted thoughts or None if no thoughts found
        """
        for pattern, group in self.patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(group).strip()
        
        return None
    
    def extract_content(self, text: str) -> str:
        """
        Extract the content without thinking sections.
        
        Args:
            text: The text to parse
            
        Returns:
            Text with thinking sections removed
        """
        for pattern, _ in self.patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text.strip()


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text (rough approximation).
    
    Args:
        text: The text to estimate
        
    Returns:
        Estimated token count
    """
    # Very simplistic token estimation (4 chars per token on average)
    return len(text) // 4 
"""
Tools system for the custom agent.
"""

import inspect
import re
import json
import requests
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from pydantic import BaseModel, Field, create_model
from datetime import datetime

class ToolParameter(BaseModel):
    """Parameter for a tool."""
    name: str = Field(..., description="Name of the parameter")
    type: str = Field(..., description="Type of the parameter")
    description: str = Field(..., description="Description of the parameter")
    required: bool = Field(True, description="Whether the parameter is required")
    
class ToolMetadata(BaseModel):
    """Metadata for a tool."""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Parameters for the tool")

class ToolOutput(BaseModel):
    """Output from a tool execution."""
    content: Any = Field(..., description="Content of the output")
    error: Optional[str] = Field(None, description="Error message if tool execution failed")

class Tool:
    """Base class for tools."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize a tool.
        
        Args:
            name: Name of the tool (defaults to class or function name)
            description: Description of the tool (defaults to docstring)
            parameters: Parameters for the tool (defaults to function signature)
        """
        self.name = name
        self.description = description
        self._parameters = parameters
        
        # Will be set during registration
        self._function = None
        self._metadata = None
        
    def __call__(self, **kwargs) -> ToolOutput:
        """
        Execute the tool.
        
        Args:
            **kwargs: Arguments for the tool
            
        Returns:
            ToolOutput with the result
        """
        if not self._function:
            raise ValueError("Tool has not been registered with a function")
            
        try:
            result = self._function(**kwargs)
            return ToolOutput(content=result)
        except Exception as e:
            return ToolOutput(content=None, error=str(e))
    
    @property
    def metadata(self) -> ToolMetadata:
        """
        Get metadata for the tool.
        
        Returns:
            ToolMetadata
        """
        if self._metadata:
            return self._metadata
            
        # Extract function signature information
        if not self._function:
            raise ValueError("Tool has not been registered with a function")
            
        func = self._function
        name = self.name or func.__name__
        description = self.description or (func.__doc__ or "").strip()
        
        # Get parameters from function signature or override
        parameters = []
        if self._parameters:
            # Use provided parameters
            for param in self._parameters:
                parameters.append(
                    ToolParameter(
                        name=param.get("name"),
                        type=param.get("type", "string"),
                        description=param.get("description", ""),
                        required=param.get("required", True)
                    )
                )
        else:
            # Extract from function signature
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            
            for param_name, param in sig.parameters.items():
                # Skip self parameter
                if param_name == "self":
                    continue
                    
                param_type = type_hints.get(param_name, Any).__name__
                has_default = param.default is not inspect.Parameter.empty
                
                # Try to extract description from docstring
                param_desc = ""
                if func.__doc__:
                    param_match = re.search(
                        fr"\s+{param_name}:\s*(.*?)(?:\n\s+\w+:|$)", 
                        func.__doc__, 
                        re.DOTALL
                    )
                    if param_match:
                        param_desc = param_match.group(1).strip()
                
                parameters.append(
                    ToolParameter(
                        name=param_name,
                        type=param_type,
                        description=param_desc,
                        required=not has_default
                    )
                )
        
        self._metadata = ToolMetadata(
            name=name,
            description=description,
            parameters=parameters
        )
        
        return self._metadata

    def to_openai_function(self) -> Dict[str, Any]:
        """
        Convert the tool to an OpenAI function schema.
        
        Returns:
            Dict representation of the tool as an OpenAI function
        """
        meta = self.metadata
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in meta.parameters:
            # Convert tool parameter type to JSON Schema type
            param_type = "string"  # Default
            if param.type in ["int", "integer", "float", "number"]:
                param_type = "number"
            elif param.type in ["bool", "boolean"]:
                param_type = "boolean"
            elif param.type in ["list", "array", "List"]:
                param_type = "array"
            elif param.type in ["dict", "object", "Dict"]:
                param_type = "object"
                
            parameters["properties"][param.name] = {
                "type": param_type,
                "description": param.description
            }
            
            if param.required:
                parameters["required"].append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": meta.name,
                "description": meta.description,
                "parameters": parameters
            }
        }
        
    def to_smolagents_tool(self) -> Any:
        """
        Convert to a smolagents tool format.
        
        Returns:
            A tool in smolagents format (if smolagents is available)
        """
        try:
            # Attempt to import smolagents, but handle gracefully if not available
            from smolagents import Tool as SmolagentsTool
            
            # Create a smolagents Tool wrapper
            class WrappedSmolagentsTool(SmolagentsTool):
                def __init__(self, original_tool):
                    self.original_tool = original_tool
                    # Map metadata to smolagents format
                    meta = original_tool.metadata
                    self.name = meta.name
                    self.description = meta.description
                    self.inputs = {}
                    
                    # Convert parameters
                    for param in meta.parameters:
                        self.inputs[param.name] = {
                            "type": param.type.lower(),
                            "description": param.description
                        }
                
                def forward(self, **kwargs):
                    # Execute the original tool
                    result = self.original_tool(**kwargs)
                    # Return content or error message
                    return result.content if result.error is None else f"Error: {result.error}"
            
            return WrappedSmolagentsTool(self)
        except ImportError:
            # If smolagents is not available, return None
            print("Warning: smolagents package not available. Cannot convert to smolagents tool.")
            return None


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[List[Dict[str, Any]]] = None
) -> Callable:
    """
    Decorator to convert a function into a Tool.
    
    Args:
        name: Name for the tool (defaults to function name)
        description: Description for the tool (defaults to docstring)
        parameters: Parameters for the tool (defaults to function signature)
        
    Returns:
        Decorated function as a Tool
    """
    def decorator(func: Callable) -> Tool:
        tool_instance = Tool(name=name, description=description, parameters=parameters)
        tool_instance._function = func
        # Generate metadata immediately
        _ = tool_instance.metadata
        return tool_instance
    
    return decorator


class FinalAnswer(Tool):
    """
    A tool for providing a final answer to the task, which terminates agent execution.
    
    This tool is used to indicate that the agent has completed its task and wants to
    provide a final answer. After calling this tool, the agent's execution loop ends.
    """
    
    name = "final_answer"
    description = "Use this tool when you have a final answer to the user's query. This will end the conversation."
    
    def __init__(self):
        """Initialize the FinalAnswer tool."""
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=[
                {
                    "name": "answer",
                    "type": "string",
                    "description": "The final answer to the user's query",
                    "required": True
                }
            ]
        )
        self._function = self.forward
    
    def forward(self, answer: str) -> str:
        """
        Process the final answer.
        
        Args:
            answer: The final answer string provided by the agent
            
        Returns:
            The final answer string
        """
        return answer


class SearchEngine(Tool):
    """
    A tool for searching the web for information.
    
    This tool uses Serper API (if API key is available) or falls back to DuckDuckGo 
    to retrieve information from the web.
    """
    
    name = "search"
    description = "Search the web for information."
    
    def __init__(self, urls: Optional[List[str]] = None, cutoff_date: Optional[datetime] = None):
        """
        Initialize the SearchEngine tool.
        
        Args:
            urls: Optional list of URLs to restrict search to
            cutoff_date: Optional cutoff date to restrict search results
        """
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=[
                {
                    "name": "query",
                    "type": "string",
                    "description": "The search query to execute",
                    "required": True
                },
                {
                    "name": "num_results",
                    "type": "integer",
                    "description": "Number of search results to return (default: 5)",
                    "required": False
                }
            ]
        )
        self._function = self.forward
        
        # Store URLs and cutoff date for filtering
        self.urls = urls
        self.cutoff_date = cutoff_date
        
        # Try to get API key from environment
        try:
            import os
            self.serper_api_key = os.environ.get("SERPER_API_KEY")
        except Exception:
            self.serper_api_key = None
    
    def _get_search_params(self, query: str) -> dict:
        """
        Get search parameters with URL and date filtering if specified.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary of search parameters
        """
        params = {
            "q": query,
            "gl": "us",
            "hl": "en",
            "autocorrect": True,
        }

        # Add date filtering if specified
        if self.cutoff_date:
            if isinstance(self.cutoff_date, datetime):
                params["q"] = f"{params['q']} after:{self.cutoff_date.strftime('%Y/%m/%d')}"

        # Add URL filtering if specified
        if self.urls:
            site_query = " OR ".join([f"site:{url}" for url in self.urls])
            params["q"] = f"{params['q']} ({site_query})"

        return params
    
    def _process_serper_results(self, results: dict, query: str) -> str:
        """
        Process search results from Serper API.
        
        Args:
            results: The search results from Serper API
            query: The original search query
            
        Returns:
            Formatted search results string
        """
        if "organic" not in results.keys():
            urls_msg = f" with URL restriction to {self.urls}" if self.urls else ""
            return f"No 'organic' results found for query: '{query}'{urls_msg}. Try a less restrictive query."

        if len(results["organic"]) == 0:
            urls_message = f" with URL restriction to {self.urls}" if self.urls else ""
            return f"No results found for '{query}'{urls_message}. Try with a more general query."

        web_snippets = []
        for idx, page in enumerate(results["organic"]):
            date_published = f"\nDate published: {page.get('date', '')}" if "date" in page else ""
            attributes = ""
            if "attributes" in page:
                for key, value in page["attributes"].items():
                    attributes += f"\n{key}: {value}"

            snippet = f"\n{page['snippet']}" if "snippet" in page else ""

            # Build the result entry
            result_entry = (
                f"{idx+1}. [{page['title']}]({page['link']}){date_published}{attributes}{snippet}"
            )

            # Add sitelinks if available
            if "sitelinks" in page and page["sitelinks"]:
                result_entry += "\nRelated links:"
                for sitelink in page["sitelinks"]:
                    result_entry += f"\n- [{sitelink['title']}]({sitelink['link']})"

            web_snippets.append(result_entry)

        return "## Search Results\n" + "\n\n".join(web_snippets)
    
    def _search_with_serper(self, query: str) -> str:
        """
        Search the web using Serper API.
        
        Args:
            query: The search query
            
        Returns:
            Search results string
        """
        import requests
        
        params = self._get_search_params(query)
        
        headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
        
        response = requests.post("https://google.serper.dev/search", json=params, headers=headers)
        
        if response.status_code == 200:
            results = response.json()
            return self._process_serper_results(results, query)
        else:
            raise ValueError(f"Error from Serper API: {response.text}")
    
    def _search_with_duckduckgo(self, query: str, num_results: int = 5) -> str:
        """
        Search the web using DuckDuckGo API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            Search results string
        """
        import requests
        
        # Apply URL filtering if specified
        if self.urls:
            site_query = " OR ".join([f"site:{url}" for url in self.urls])
            query = f"{query} ({site_query})"
            
        # Apply date filtering if specified (limited support in DuckDuckGo)
        if self.cutoff_date:
            if isinstance(self.cutoff_date, datetime):
                query = f"{query} after:{self.cutoff_date.year}"
        
        # Use DuckDuckGo API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        results = []
        
        # Process Abstract if available
        if data.get('Abstract'):
            results.append({
                'title': data.get('Heading', 'Abstract'),
                'snippet': data['Abstract'],
                'url': data.get('AbstractURL', '')
            })
        
        # Process RelatedTopics
        for topic in data.get('RelatedTopics', [])[:num_results]:
            if 'Text' in topic and 'FirstURL' in topic:
                results.append({
                    'title': topic.get('Text', '').split(' - ')[0],
                    'snippet': topic.get('Text', ''),
                    'url': topic.get('FirstURL', '')
                })
        
        if not results:
            return "No search results found."
        
        # Format results
        output = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results[:num_results], 1):
            output += f"{i}. {result['title']}\n"
            output += f"   {result['snippet']}\n"
            output += f"   URL: {result['url']}\n\n"
        
        return output
    
    def forward(self, query: str, num_results: int = 5) -> str:
        """
        Execute a web search using available API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            String containing search results
        """
        try:
            # Try to use Serper API if key is available
            if self.serper_api_key:
                return self._search_with_serper(query)
            else:
                # Fall back to DuckDuckGo
                return self._search_with_duckduckgo(query, num_results)
            
        except Exception as e:
            return f"Error executing search: {str(e)}"


class WebScraper(Tool):
    """
    A tool for scraping content from web pages.
    
    This tool fetches and extracts clean text content from web pages using
    BeautifulSoup if available, or falls back to regex-based extraction.
    """
    
    name = "scrape"
    description = "Scrape text content from a web page."
    
    def __init__(self):
        """Initialize the WebScraper tool."""
        super().__init__(
            name=self.name,
            description=self.description,
            parameters=[
                {
                    "name": "url",
                    "type": "string",
                    "description": "URL of the web page to scrape",
                    "required": True
                },
                {
                    "name": "format",
                    "type": "string",
                    "description": "Output format: 'text' or 'markdown' (default: 'text')",
                    "required": False
                }
            ]
        )
        self._function = self.forward
        
        # Check if BeautifulSoup is available
        try:
            import bs4
            self.bs4_available = True
        except ImportError:
            self.bs4_available = False
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the text by removing extra whitespace.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_with_bs4(self, html: str, output_format: str = 'text') -> str:
        """
        Extract content using BeautifulSoup.
        
        Args:
            html: HTML content
            output_format: Output format ('text' or 'markdown')
            
        Returns:
            Extracted content
        """
        from bs4 import BeautifulSoup
        
        # Parse the HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, and other undesirable tags
        for tag in soup(["script", "style", "meta", "noscript", "svg", "iframe"]):
            tag.decompose()
        
        # Get the text
        if output_format == 'markdown':
            # Try to import markdownify if available
            try:
                from markdownify import markdownify
                text = markdownify(str(soup))
            except ImportError:
                # Fall back to simple text with paragraph breaks
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if paragraphs:
                    text = "\n\n".join([p.get_text() for p in paragraphs])
                else:
                    text = soup.get_text()
                    text = self._clean_text(text)
        else:
            # Simple text extraction
            text = soup.get_text()
            text = self._clean_text(text)
        
        return text
    
    def _extract_with_regex(self, html: str) -> str:
        """
        Extract content using regex.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted content
        """
        import re
        
        # Remove scripts, styles, and HTML tags
        html = re.sub(r'<script.*?</script>', ' ', html, flags=re.DOTALL)
        html = re.sub(r'<style.*?</style>', ' ', html, flags=re.DOTALL)
        html = re.sub(r'<head.*?</head>', ' ', html, flags=re.DOTALL)
        html = re.sub(r'<nav.*?</nav>', ' ', html, flags=re.DOTALL)
        html = re.sub(r'<footer.*?</footer>', ' ', html, flags=re.DOTALL)
        
        # Replace common elements with line breaks
        html = re.sub(r'</?(?:div|p|br|h[1-6]|li)[^>]*>', '\n', html)
        
        # Remove remaining HTML tags
        html = re.sub(r'<[^>]*>', ' ', html)
        
        # Remove extra whitespace
        text = self._clean_text(html)
        
        return text
    
    def forward(self, url: str, format: str = 'text') -> str:
        """
        Scrape content from a web page.
        
        Args:
            url: URL of the web page to scrape
            format: Output format ('text' or 'markdown')
            
        Returns:
            Extracted text content from the web page
        """
        try:
            # Check if URL starts with http
            if not url.startswith('http'):
                url = 'https://' + url
                
            # Fetch page content
            import requests
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            }, timeout=10)
            response.raise_for_status()
            
            # Use BeautifulSoup if available
            if self.bs4_available:
                text = self._extract_with_bs4(response.text, format)
            else:
                text = self._extract_with_regex(response.text)
            
            # Limit to 8000 characters to prevent too much output
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"
            
            # Add a header with metadata
            header = f"# Content from {url}\n\n"
            result = header + text
            
            return result
        
        except Exception as e:
            return f"Error scraping URL {url}: {str(e)}"


def add_final_answer_to_agent(agent: Any) -> Any:
    """
    Add the FinalAnswer tool to an agent.
    
    Args:
        agent: The agent to add the tool to
        
    Returns:
        The agent with the FinalAnswer tool added
    """
    # Create a new FinalAnswer tool
    final_answer_tool = FinalAnswer()
    
    # Add the tool to the agent's tools list
    agent.tools.append(final_answer_tool)
    
    return agent


def get_default_tools() -> List[Tool]:
    """
    Get a list of default tools.
    
    Returns:
        List of default tools
    """
    return [
        FinalAnswer(),
        SearchEngine(),
        WebScraper()
    ] 
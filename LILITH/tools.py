"""Tool calling and function execution framework for AI agents.

Enables LILITH to use tools like code execution, web search, calculators, etc.
"""
from __future__ import annotations

from typing import Callable, Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import inspect


class ToolType(Enum):
    """Types of tools available."""
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class Tool:
    """A tool that the AI can use."""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    tool_type: ToolType = ToolType.CUSTOM
    returns: Optional[str] = None

    def to_schema(self) -> Dict:
        """Convert to OpenAI function calling schema."""
        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param in self.parameters:
            param_schema = {
                "type": param.type,
                "description": param.description,
            }

            if param.enum:
                param_schema["enum"] = param.enum

            parameters_schema["properties"][param.name] = param_schema

            if param.required:
                parameters_schema["required"].append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters_schema,
        }

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        # Validate required parameters
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")

            # Set default if not provided
            if param.name not in kwargs and param.default is not None:
                kwargs[param.name] = param.default

        # Execute function
        return self.function(**kwargs)


@dataclass
class ToolCall:
    """A request to call a tool."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result from executing a tool."""
    call_id: Optional[str]
    tool_name: str
    result: Any
    error: Optional[str] = None
    success: bool = True


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tool_type: ToolType = ToolType.CUSTOM,
    ) -> Tool:
        """Register a function as a tool.

        Automatically extracts parameters from function signature.
        """
        if name is None:
            name = func.__name__

        if description is None:
            description = func.__doc__ or f"Execute {name}"

        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default

            if param.annotation != inspect.Parameter.empty:
                # Try to infer type
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation in (list, List):
                    param_type = "array"
                elif param.annotation in (dict, Dict):
                    param_type = "object"

            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter {param_name}",
                required=required,
                default=default,
            ))

        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=func,
            tool_type=tool_type,
        )

        self.register(tool)
        return tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self.tools.values())

    def get_schemas(self) -> List[Dict]:
        """Get all tool schemas for function calling."""
        return [tool.to_schema() for tool in self.tools.values()]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)

        if not tool:
            return ToolResult(
                call_id=None,
                tool_name=tool_name,
                result=None,
                error=f"Tool not found: {tool_name}",
                success=False,
            )

        try:
            result = tool.execute(**kwargs)
            return ToolResult(
                call_id=None,
                tool_name=tool_name,
                result=result,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                call_id=None,
                tool_name=tool_name,
                result=None,
                error=str(e),
                success=False,
            )


# Built-in tools
def create_default_tools() -> ToolRegistry:
    """Create registry with default tools."""
    registry = ToolRegistry()

    # Calculator tool
    def calculator(expression: str) -> float:
        """Evaluate a mathematical expression safely."""
        try:
            # Safe eval (limited to math operations)
            import ast
            import operator

            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }

            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(node)

            return eval_expr(ast.parse(expression, mode='eval').body)
        except Exception as e:
            return f"Error: {str(e)}"

    registry.register_function(
        calculator,
        description="Evaluate mathematical expressions",
        tool_type=ToolType.CALCULATOR,
    )

    # Code execution tool (using SHAMSHEL)
    def execute_python(code: str) -> str:
        """Execute Python code safely in a sandbox."""
        try:
            from SHAMSHEL import SandboxRunner

            runner = SandboxRunner(timeout_sec=5, max_memory_mb=100)
            result = runner.run_python(code)

            if result.success:
                return result.stdout or "Code executed successfully"
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error: {str(e)}"

    registry.register_function(
        execute_python,
        name="execute_python",
        description="Execute Python code in a secure sandbox",
        tool_type=ToolType.CODE_EXECUTION,
    )

    return registry


def parse_tool_calls_from_text(text: str) -> List[ToolCall]:
    """Parse tool calls from model output.

    Expected format:
    <tool_call>
    {
        "name": "calculator",
        "arguments": {"expression": "2 + 2"}
    }
    </tool_call>
    """
    import re

    tool_calls = []
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match.strip())
            tool_calls.append(ToolCall(
                tool_name=data["name"],
                arguments=data.get("arguments", {}),
            ))
        except json.JSONDecodeError:
            continue

    return tool_calls


def format_tool_result_for_prompt(result: ToolResult) -> str:
    """Format tool result to inject back into prompt."""
    if result.success:
        return f"""<tool_result tool="{result.tool_name}">
{result.result}
</tool_result>"""
    else:
        return f"""<tool_result tool="{result.tool_name}" error="true">
{result.error}
</tool_result>"""


__all__ = [
    "ToolType",
    "ToolParameter",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "create_default_tools",
    "parse_tool_calls_from_text",
    "format_tool_result_for_prompt",
]

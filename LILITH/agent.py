"""Autonomous agent system for LILITH.

This module enables LILITH to act as an autonomous agent:
- Multi-turn conversations
- Tool use and function calling
- Reasoning and planning
- Memory and context management
"""
from __future__ import annotations

from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
import json

import torch

from .model import GPTDecoder
from .sampling import sample_with_strategy
from .tools import ToolRegistry, ToolCall, parse_tool_calls_from_text, format_tool_result_for_prompt


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List] = None


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_turns: int = 10
    max_tool_calls_per_turn: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    system_prompt: Optional[str] = None
    enable_tools: bool = True
    enable_thinking: bool = True  # Show reasoning process


class LILITHAgent:
    """Autonomous conversational agent powered by LILITH."""

    def __init__(
        self,
        model: GPTDecoder,
        tokenizer,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[AgentConfig] = None,
    ):
        """Initialize the agent.

        Args:
            model: The LILITH model
            tokenizer: Tokenizer for encoding/decoding
            tool_registry: Registry of available tools
            config: Agent configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()

        # Conversation history
        self.messages: List[Message] = []

        # Add system prompt if provided
        if self.config.system_prompt:
            self.add_message("system", self.config.system_prompt)

        # Device
        self.device = next(model.parameters()).device
        self.model.eval()

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append(Message(role=role, content=content))

    def format_conversation(self, include_tools: bool = False) -> str:
        """Format conversation history as a prompt.

        Args:
            include_tools: Whether to include tool schemas

        Returns:
            Formatted prompt string
        """
        parts = []

        # Add tool schemas if enabled
        if include_tools and self.tool_registry:
            tools_desc = "You have access to the following tools:\n\n"
            for tool in self.tool_registry.list_tools():
                schema = tool.to_schema()
                tools_desc += f"- {schema['name']}: {schema['description']}\n"

            tools_desc += """\nTo use a tool, output:
<tool_call>
{
    "name": "tool_name",
    "arguments": {"param": "value"}
}
</tool_call>
"""
            parts.append(tools_desc)

        # Format messages
        for msg in self.messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}\n")
            elif msg.role == "user":
                parts.append(f"\nHuman: {msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"\nAssistant: {msg.content}\n")
            elif msg.role == "tool":
                parts.append(f"\n{msg.content}\n")

        # Add prompt for next assistant response
        parts.append("\nAssistant:")

        return "".join(parts)

    def generate_response(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate a response from the model.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        # Format prompt
        prompt_text = self.format_conversation(include_tools=self.config.enable_tools)

        # Encode
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt_text)], device=self.device)

        # Generate
        with torch.no_grad():
            generated = sample_with_strategy(
                self.model,
                prompt_ids,
                max_new_tokens=max_tokens,
                strategy="top_p",
                temperature=temperature,
                top_p=top_p,
            )

        # Decode
        full_text = self.tokenizer.decode(generated[0].tolist())

        # Extract only the new response (after the prompt)
        response = full_text[len(prompt_text):].strip()

        return response

    def execute_tool_calls(self, text: str) -> List[str]:
        """Execute any tool calls in the generated text.

        Args:
            text: Generated text potentially containing tool calls

        Returns:
            List of tool result strings
        """
        if not self.tool_registry:
            return []

        tool_calls = parse_tool_calls_from_text(text)
        results = []

        for call in tool_calls[:self.config.max_tool_calls_per_turn]:
            result = self.tool_registry.execute(call.tool_name, **call.arguments)
            result_str = format_tool_result_for_prompt(result)
            results.append(result_str)

        return results

    def chat(self, user_message: str, verbose: bool = False) -> str:
        """Chat with the agent (single turn).

        Args:
            user_message: User's message
            verbose: Whether to show thinking process

        Returns:
            Agent's response
        """
        # Add user message
        self.add_message("user", user_message)

        # Generate response
        response = self.generate_response()

        if verbose and self.config.enable_thinking:
            print(f"[THINKING] Generated: {response[:200]}...")

        # Check for tool calls
        if self.config.enable_tools:
            tool_results = self.execute_tool_calls(response)

            if tool_results:
                if verbose:
                    print(f"[TOOLS] Executed {len(tool_results)} tool(s)")

                # Add tool results to conversation
                for result in tool_results:
                    self.add_message("tool", result)

                # Generate final response incorporating tool results
                response = self.generate_response()

        # Add assistant response to history
        self.add_message("assistant", response)

        return response

    def autonomous_loop(
        self,
        task: str,
        max_iterations: int = 10,
        verbose: bool = True,
    ) -> List[str]:
        """Run autonomous agent loop to accomplish a task.

        Args:
            task: Task description
            max_iterations: Maximum iterations
            verbose: Whether to print progress

        Returns:
            List of all responses
        """
        if verbose:
            print(f"🤖 Starting autonomous task: {task}\n")
            print("=" * 60)

        # Set task as initial message
        self.add_message("user", f"Task: {task}\n\nComplete this task step by step. Use tools when needed.")

        responses = []

        for i in range(max_iterations):
            if verbose:
                print(f"\n[Iteration {i+1}/{max_iterations}]")

            # Generate response
            response = self.generate_response()

            if verbose:
                print(f"Agent: {response}\n")

            responses.append(response)

            # Check for tool calls
            if self.config.enable_tools:
                tool_results = self.execute_tool_calls(response)

                if tool_results:
                    if verbose:
                        print(f"🔧 Executed {len(tool_results)} tool(s):")
                        for result in tool_results:
                            print(f"  {result[:100]}...")

                    for result in tool_results:
                        self.add_message("tool", result)
                else:
                    # No more tools to call, task likely complete
                    if "task complete" in response.lower() or "finished" in response.lower():
                        if verbose:
                            print("\n✅ Task completed!")
                        break

            # Add response to history
            self.add_message("assistant", response)

        if verbose:
            print("\n" + "=" * 60)
            print(f"🏁 Completed in {len(responses)} iterations")

        return responses

    def reset(self, keep_system: bool = True):
        """Reset conversation history.

        Args:
            keep_system: Whether to keep system prompt
        """
        if keep_system and self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def save_conversation(self, path: str):
        """Save conversation to file."""
        conversation_data = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in self.messages
        ]

        with open(path, 'w') as f:
            json.dump(conversation_data, f, indent=2)

    def load_conversation(self, path: str):
        """Load conversation from file."""
        with open(path, 'r') as f:
            conversation_data = json.load(f)

        self.messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in conversation_data
        ]


def create_assistant_agent(
    model: GPTDecoder,
    tokenizer,
    personality: str = "helpful",
) -> LILITHAgent:
    """Create a helpful assistant agent.

    Args:
        model: LILITH model
        tokenizer: Tokenizer
        personality: Assistant personality type

    Returns:
        Configured agent
    """
    from .instruction_data import SYSTEM_PROMPTS
    from .tools import create_default_tools

    personalities = {
        "helpful": SYSTEM_PROMPTS["helpful_assistant"],
        "coding": SYSTEM_PROMPTS["coding_assistant"],
        "creative": SYSTEM_PROMPTS["creative_writer"],
        "teacher": SYSTEM_PROMPTS["teacher"],
        "researcher": SYSTEM_PROMPTS["researcher"],
    }

    config = AgentConfig(
        system_prompt=personalities.get(personality, personalities["helpful"]),
        enable_tools=True,
        enable_thinking=True,
        temperature=0.7,
        top_p=0.9,
    )

    tool_registry = create_default_tools()

    return LILITHAgent(
        model=model,
        tokenizer=tokenizer,
        tool_registry=tool_registry,
        config=config,
    )


__all__ = [
    "Message",
    "AgentConfig",
    "LILITHAgent",
    "create_assistant_agent",
]

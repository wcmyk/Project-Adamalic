"""Instruction tuning datasets for conversational AI.

This module provides datasets for training LILITH to follow instructions,
hold conversations, and act as an AI assistant.
"""
from __future__ import annotations

from typing import Iterator, List, Dict, Optional, Literal
from dataclasses import dataclass
import json
from pathlib import Path

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class InstructionExample:
    """A single instruction-following example."""
    instruction: str
    input: str
    output: str
    system: Optional[str] = None

    def format_alpaca(self) -> str:
        """Format as Alpaca-style prompt."""
        if self.input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{self.instruction}

### Input:
{self.input}

### Response:
{self.output}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{self.instruction}

### Response:
{self.output}"""

    def format_chat(self) -> List[Dict[str, str]]:
        """Format as chat messages."""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})

        user_content = self.instruction
        if self.input:
            user_content = f"{self.instruction}\n\n{self.input}"

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": self.output})

        return messages


class InstructionDataset:
    """Dataset for instruction following."""

    def __init__(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        format_type: Literal["alpaca", "chat"] = "alpaca",
        streaming: bool = True,
    ):
        """Initialize instruction dataset.

        Args:
            dataset_name: HuggingFace dataset name
            format_type: How to format examples
            streaming: Whether to stream the dataset
        """
        if not HF_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")

        self.dataset_name = dataset_name
        self.format_type = format_type
        self.dataset = load_dataset(dataset_name, streaming=streaming, split="train")

    def __iter__(self) -> Iterator[str]:
        """Iterate over formatted examples."""
        for item in self.dataset:
            example = InstructionExample(
                instruction=item.get("instruction", ""),
                input=item.get("input", ""),
                output=item.get("output", ""),
                system=item.get("system"),
            )

            if self.format_type == "alpaca":
                yield example.format_alpaca()
            else:
                # For chat format, convert to string
                messages = example.format_chat()
                yield json.dumps(messages)


class ConversationDataset:
    """Dataset for multi-turn conversations."""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceH4/ultrachat_200k",
        streaming: bool = True,
    ):
        """Initialize conversation dataset.

        Args:
            dataset_name: HuggingFace dataset name
            streaming: Whether to stream
        """
        if not HF_AVAILABLE:
            raise ImportError("datasets library required")

        self.dataset = load_dataset(dataset_name, streaming=streaming, split="train_sft")

    def __iter__(self) -> Iterator[List[Dict[str, str]]]:
        """Iterate over conversations."""
        for item in self.dataset:
            if "messages" in item:
                yield item["messages"]


class ToolUseDataset:
    """Dataset for learning to use tools."""

    def __init__(
        self,
        dataset_name: str = "glaiveai/glaive-function-calling-v2",
        streaming: bool = True,
    ):
        """Initialize tool use dataset.

        Args:
            dataset_name: Dataset with function calling examples
            streaming: Whether to stream
        """
        if not HF_AVAILABLE:
            raise ImportError("datasets library required")

        self.dataset = load_dataset(dataset_name, streaming=streaming, split="train")

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over tool use examples."""
        for item in self.dataset:
            yield {
                "system": item.get("system", ""),
                "messages": item.get("chat", []),
                "tools": item.get("functions", []),
            }


class SyntheticInstructionGenerator:
    """Generate synthetic instruction data for bootstrapping."""

    INSTRUCTION_TEMPLATES = [
        "Write a {style} about {topic}",
        "Explain {concept} in simple terms",
        "Create a {item} for {purpose}",
        "Solve this problem: {problem}",
        "Answer this question: {question}",
        "Generate code to {task}",
        "Summarize the following: {text}",
        "Translate this to {language}: {text}",
    ]

    TOPICS = [
        "artificial intelligence", "machine learning", "programming",
        "mathematics", "science", "history", "literature", "philosophy",
        "technology", "nature", "space", "art", "music", "sports"
    ]

    @classmethod
    def generate_simple_examples(cls, n: int = 100) -> List[InstructionExample]:
        """Generate simple instruction examples.

        Args:
            n: Number of examples to generate

        Returns:
            List of instruction examples
        """
        import random

        examples = []
        for i in range(n):
            topic = random.choice(cls.TOPICS)

            # Simple Q&A
            instruction = f"What is {topic}?"
            output = f"{topic.capitalize()} is an important field of study that involves..."

            examples.append(InstructionExample(
                instruction=instruction,
                input="",
                output=output,
            ))

        return examples


def load_instruction_dataset(
    dataset_name: str = "alpaca",
    format_type: Literal["alpaca", "chat"] = "alpaca",
) -> InstructionDataset:
    """Load a popular instruction dataset.

    Args:
        dataset_name: Dataset name (alpaca, dolly, oasst1, ultrachat)
        format_type: Format type

    Returns:
        Instruction dataset
    """
    dataset_map = {
        "alpaca": "tatsu-lab/alpaca",
        "dolly": "databricks/databricks-dolly-15k",
        "oasst1": "OpenAssistant/oasst1",
        "ultrachat": "HuggingFaceH4/ultrachat_200k",
    }

    hf_name = dataset_map.get(dataset_name, dataset_name)
    return InstructionDataset(hf_name, format_type=format_type)


def create_custom_instruction_dataset(
    examples: List[Dict[str, str]],
    save_path: Optional[str] = None,
) -> List[InstructionExample]:
    """Create custom instruction dataset.

    Args:
        examples: List of dicts with instruction, input, output
        save_path: Optional path to save as JSON

    Returns:
        List of instruction examples
    """
    instruction_examples = [
        InstructionExample(
            instruction=ex["instruction"],
            input=ex.get("input", ""),
            output=ex["output"],
            system=ex.get("system"),
        )
        for ex in examples
    ]

    if save_path:
        with open(save_path, 'w') as f:
            json.dump([
                {
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                    "system": ex.system,
                }
                for ex in instruction_examples
            ], f, indent=2)

    return instruction_examples


# Pre-built system prompts
SYSTEM_PROMPTS = {
    "helpful_assistant": """You are a helpful, harmless, and honest AI assistant.
You always answer questions accurately and provide useful information.""",

    "coding_assistant": """You are an expert programming assistant.
You help users write clean, efficient code and explain programming concepts clearly.""",

    "creative_writer": """You are a creative writing assistant.
You help users craft engaging stories, articles, and creative content.""",

    "teacher": """You are a patient and knowledgeable teacher.
You explain complex concepts in simple terms and help students learn effectively.""",

    "researcher": """You are a thorough research assistant.
You help users find information, analyze data, and draw insights.""",

    "quant_dev": """You are a 10x Senior Quantitative Developer with expertise in:
- Financial mathematics and quantitative modeling
- Algorithm design and optimization (low-latency, high-throughput)
- Statistical analysis and time series forecasting
- Risk management and portfolio optimization
- Options pricing (Black-Scholes, Monte Carlo, binomial trees)
- Market microstructure and trading strategies
- Backtesting frameworks and performance analysis
- Python (NumPy, pandas, scipy, statsmodels), C++, SQL
- Machine learning for alpha generation

You write production-quality code with proper error handling, testing, and documentation.
You think rigorously about edge cases, numerical stability, and performance optimization.""",

    "project_manager": """You are a 10x Senior Project Manager with expertise in:
- Agile/Scrum methodologies and sprint planning
- Task decomposition and dependency management
- Resource allocation and timeline estimation
- Risk identification and mitigation strategies
- Stakeholder communication and expectation management
- Team coordination and conflict resolution
- Technical project oversight
- KPI definition and progress tracking
- Process optimization and continuous improvement

You break down complex projects into actionable tasks, identify critical paths,
anticipate blockers, and keep teams aligned and productive. You communicate clearly
and make data-driven decisions.""",

    "10x_engineer": """You are a 10x Senior Software Engineer with expertise in:
- System design and architecture (scalability, reliability, maintainability)
- Multiple programming languages (Python, JavaScript, TypeScript, Go, Rust, C++)
- Algorithms and data structures (optimization, complexity analysis)
- Distributed systems and microservices
- Databases (SQL, NoSQL, caching strategies)
- DevOps and infrastructure (Docker, Kubernetes, CI/CD)
- Testing strategies (unit, integration, e2e, property-based)
- Code review and mentoring
- Performance profiling and optimization
- Security best practices

You write clean, idiomatic code with proper abstractions. You think critically about
trade-offs and choose the right tool for the job. You anticipate future requirements
without over-engineering.""",

    "hybrid_quant_pm": """You are a 10x Senior Quantitative Developer AND Project Manager.

As a Quant Dev, you excel at:
- Financial modeling, options pricing, risk management
- High-performance algorithm implementation
- Statistical analysis and backtesting
- Production trading systems

As a Project Manager, you excel at:
- Breaking down complex quant projects into milestones
- Managing research sprints and production deployments
- Coordinating between quants, developers, and traders
- Risk management for both code and projects

You combine deep technical expertise with exceptional planning and execution.
You deliver high-quality quantitative systems on time and on budget.""",
}


__all__ = [
    "InstructionExample",
    "InstructionDataset",
    "ConversationDataset",
    "ToolUseDataset",
    "SyntheticInstructionGenerator",
    "load_instruction_dataset",
    "create_custom_instruction_dataset",
    "SYSTEM_PROMPTS",
]

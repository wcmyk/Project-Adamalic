```markdown
# 🤖 LILITH Phase 3: The Agent Awakening

**Transform LILITH into a conversational AI assistant like Claude**

Phase 3 enables LILITH to:
- 💬 Hold natural conversations
- 🛠️ Use tools (code execution, calculator, web search)
- 🧠 Think and reason autonomously
- 🎯 Complete complex tasks step-by-step
- 📚 Learn from human preferences (RLHF)

---

## 🎯 What is Phase 3?

Phase 3 adds **conversational AI capabilities** to LILITH, making it act like me (Claude) rather than just completing text.

### Key Transformations

| Capability | Phase 2 | Phase 3 | Like Me? |
|------------|---------|---------|----------|
| **Text Completion** | ✅ | ✅ | Partial |
| **Conversations** | ❌ | ✅ | ✅ |
| **Tool Use** | ❌ | ✅ | ✅ |
| **Reasoning** | ❌ | ✅ | ✅ |
| **Task Completion** | ❌ | ✅ | ✅ |
| **Learning from Feedback** | ❌ | ✅ | ✅ |

---

## 📦 New Modules

### 1. **`instruction_data.py`** - Instruction Following

Train LILITH to follow instructions like:
- "Write a Python function to..."
- "Explain quantum physics in simple terms"
- "Help me debug this code"

```python
from LILITH.instruction_data import load_instruction_dataset

# Load Alpaca, Dolly, or other instruction datasets
dataset = load_instruction_dataset("alpaca")

for example in dataset:
    print(example)  # Pre-formatted instruction + response pairs
```

**Supported datasets:**
- Alpaca (52K instruction examples)
- Dolly (15K instruction examples)
- OpenAssistant (161K conversations)
- UltraChat (200K conversations)

### 2. **`tools.py`** - Tool & Function Calling

Enable LILITH to use tools like I do:

```python
from LILITH.tools import create_default_tools, ToolRegistry

# Create tool registry
tools = create_default_tools()

# Built-in tools:
# - calculator: Math operations
# - execute_python: Run code in SHAMSHEL sandbox

# Add custom tool
def web_search(query: str) -> str:
    """Search the web."""
    # Your implementation
    return f"Results for: {query}"

tools.register_function(web_search, description="Search the web")

# Execute tool
result = tools.execute("calculator", expression="2 + 2")
print(result.result)  # 4.0
```

### 3. **`agent.py`** - Autonomous Agent System

The core that makes LILITH autonomous:

```python
from LILITH.agent import create_assistant_agent

# Create agent
agent = create_assistant_agent(model, tokenizer, personality="helpful")

# Simple chat
response = agent.chat("What is machine learning?")
print(response)

# Autonomous task completion
responses = agent.autonomous_loop(
    task="Calculate fibonacci numbers and find the sum of even ones",
    max_iterations=10
)
```

**Agent capabilities:**
- Multi-turn conversations with memory
- Tool use (automatically calls tools when needed)
- Reasoning (thinks through problems step-by-step)
- Task decomposition (breaks complex tasks into steps)

### 4. **`rlhf.py`** - Learn from Human Feedback

Align LILITH with human preferences:

```python
from LILITH.rlhf import train_reward_model, PreferencePair

# Create preference pairs
pairs = [
    PreferencePair(
        prompt="Explain AI",
        chosen="AI is a field of computer science...",  # Good response
        rejected="idk lol",  # Bad response
    ),
    # ... more pairs
]

# Train reward model
reward_model = train_reward_model(model, pairs, tokenizer)

# Use reward model to score responses
reward = reward_model.score("Explain AI", "AI is...")
```

---

## 🚀 Quick Start

### Step 1: Train Conversational Model

```bash
# Train on Alpaca dataset (52K instructions)
python train_conversational.py --dataset alpaca --model_size small

# Train on multiple datasets
python train_conversational.py \
  --datasets alpaca dolly ultrachat \
  --model_size medium \
  --max_steps 20000

# Fine-tune existing model
python train_conversational.py \
  --base_model checkpoints/wikipedia/final_model.pt \
  --dataset alpaca
```

### Step 2: Create an Agent

```python
from LILITH import load_checkpoint
from LILITH.agent import create_assistant_agent

# Load trained model
model, tokenizer, _ = load_checkpoint(
    "checkpoints/conversational/conversational_model.pt"
)

# Create agent
agent = create_assistant_agent(model, tokenizer)

# Chat!
while True:
    user_input = input("You: ")
    response = agent.chat(user_input)
    print(f"Agent: {response}")
```

### Step 3: Use Tools

```python
# Agent automatically uses tools when needed
agent.chat("What is 1234 * 5678?")
# → Uses calculator tool

agent.chat("Write Python code to sort a list")
# → Uses code execution tool

agent.chat("Calculate fibonacci numbers up to 10")
# → Uses code execution tool
```

---

## 💬 Conversation Examples

### Example 1: Multi-turn Dialogue

```
You: Hello! I'm working on a Python project.
Agent: Hello! I'd be happy to help with your Python project. What are you working on?

You: I need to read data from a CSV file.
Agent: Sure! You can use Python's csv module or pandas. Here's an example with pandas:

import pandas as pd
df = pd.read_csv('your_file.csv')

Would you like me to explain more or help with specific data processing?

You: What if the CSV has missing values?
Agent: Great question! You can handle missing values in pandas using:
1. df.dropna() - Remove rows with missing values
2. df.fillna(value) - Fill missing values with a specific value
3. df.interpolate() - Interpolate missing values

Which approach fits your needs?
```

### Example 2: Tool Use

```
You: Calculate the sum of squares of numbers from 1 to 100.
Agent: I'll write and execute code to calculate that.

<tool_call>
{
  "name": "execute_python",
  "arguments": {
    "code": "result = sum(i**2 for i in range(1, 101))\nprint(result)"
  }
}
</tool_call>

<tool_result tool="execute_python">
338350
</tool_result>

The sum of squares of numbers from 1 to 100 is 338,350.
```

### Example 3: Autonomous Task

```
You: Complete this task: Find the 10th fibonacci number,
     then check if it's prime.

Agent (Iteration 1): Let me break this down:
1. Calculate the 10th Fibonacci number
2. Check if it's prime

Starting with step 1...

<tool_call>
{
  "name": "execute_python",
  "arguments": {
    "code": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a\n\nresult = fib(10)\nprint(f'10th Fibonacci: {result}')"
  }
}
</tool_call>

Agent (Iteration 2): The 10th Fibonacci number is 55.
Now checking if 55 is prime...

<tool_call>
{
  "name": "execute_python",
  "arguments": {
    "code": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\n\nprint(f'55 is prime: {is_prime(55)}')"
  }
}
</tool_call>

Agent: Task complete! The 10th Fibonacci number is 55,
and it is NOT prime (divisible by 5 and 11).
```

---

## 🎓 Training Pipeline

### Instruction Tuning (Make it helpful)

```python
python train_conversational.py \
  --datasets alpaca dolly \
  --model_size small \
  --batch_size 16 \
  --max_steps 10000
```

**What this does:**
- Trains LILITH to follow instructions
- Learns conversational patterns
- Understands question-answer format
- ~2-4 hours on RTX 3090

### Reward Modeling (Make it better)

```python
from LILITH.rlhf import train_reward_model, PreferencePair

# Collect human preferences
pairs = create_preference_pairs()  # Your preferences

# Train reward model
reward_model = train_reward_model(model, pairs, tokenizer)
```

**What this does:**
- Learns what responses humans prefer
- Understands helpful vs harmful
- Aligns with human values

### PPO Fine-tuning (Reinforce good behavior)

```python
from LILITH.rlhf import PPOTrainer

# Create PPO trainer
ppo = PPOTrainer(model, reward_model)

# Train to maximize rewards
ppo.train(prompts, num_epochs=3)
```

**What this does:**
- Optimizes model to maximize reward
- Reinforces helpful, harmless, honest behavior
- Final alignment step

---

## 🛠️ Built-in Tools

### Calculator
```python
agent.chat("What is (123 + 456) * 789?")
# → Uses calculator tool
```

### Code Execution (SHAMSHEL)
```python
agent.chat("Write Python code to reverse a string")
# → Generates and executes code safely
```

### Custom Tools

```python
from LILITH.tools import ToolRegistry, Tool, ToolParameter

registry = ToolRegistry()

# Define custom tool
def translate(text: str, language: str) -> str:
    """Translate text to another language."""
    # Your implementation
    return translated_text

# Register it
registry.register_function(
    translate,
    description="Translate text to another language"
)

# Agent can now use it
agent.tool_registry = registry
agent.chat("Translate 'hello' to Spanish")
```

---

## 🎯 Personality Presets

```python
# Helpful assistant (default)
agent = create_assistant_agent(model, tokenizer, personality="helpful")

# Coding assistant
agent = create_assistant_agent(model, tokenizer, personality="coding")

# Creative writer
agent = create_assistant_agent(model, tokenizer, personality="creative")

# Teacher
agent = create_assistant_agent(model, tokenizer, personality="teacher")

# Researcher
agent = create_assistant_agent(model, tokenizer, personality="researcher")
```

Each personality has optimized system prompts and behaviors.

---

## 📊 Training Costs

| Task | Dataset Size | Training Time | GPU | Cost (Cloud) |
|------|--------------|---------------|-----|--------------|
| **Instruction Tuning (Small)** | 52K examples | 2-4 hours | RTX 3090 | $10-20 |
| **Instruction Tuning (Medium)** | 200K examples | 8-12 hours | A100 | $50-100 |
| **Reward Modeling** | 10K pairs | 1-2 hours | RTX 3090 | $5-10 |
| **PPO Fine-tuning** | Varies | 4-8 hours | A100 | $30-60 |

**Total to create conversational assistant:**
Small model: **~$50-100**
Medium model: **~$200-300**

---

## 🎮 Try It Now

```bash
# 1. Quick test with small model
python train_conversational.py --model_size small --max_steps 1000

# 2. Run agent demo
python examples/agent_demo.py

# 3. Start interactive chat
python -c "
from LILITH import load_checkpoint
from LILITH.agent import create_assistant_agent

model, tok, _ = load_checkpoint('checkpoints/conversational/conversational_model.pt')
agent = create_assistant_agent(model, tok)

while True:
    msg = input('You: ')
    print(f'Agent: {agent.chat(msg)}')
"
```

---

## 🌟 What Makes This Special?

Unlike basic LLMs, Phase 3 LILITH:

✅ **Holds conversations** - Multi-turn dialogue with context
✅ **Uses tools** - Can execute code, do math, search
✅ **Thinks autonomously** - Breaks down complex tasks
✅ **Learns preferences** - RLHF for alignment
✅ **Customizable personality** - Different modes for different uses
✅ **Production-ready** - Complete agent framework

**This is a REAL AI assistant, not just text completion!**

---

## 🚀 Next Steps

1. **Train on your data** - Add custom instruction datasets
2. **Create custom tools** - Web search, database access, APIs
3. **Collect preferences** - Build your own reward model
4. **Deploy** - Use with FastAPI server from Phase 2
5. **Scale up** - Train larger models for better quality

---

## 🎓 Learning Resources

**How Instruction Tuning Works:**
- Start with base LLM (Wikipedia-trained)
- Fine-tune on instruction-response pairs
- Model learns to follow commands
- ~10K examples for basic capability

**How RLHF Works:**
- Collect human preferences (A vs B comparisons)
- Train reward model to predict preferences
- Use PPO to optimize policy for rewards
- Results in aligned, helpful assistant

**How Tool Use Works:**
- Model generates special tokens: `<tool_call>...</tool_call>`
- Parser extracts tool name and arguments
- Execute tool in sandbox
- Inject result back into conversation
- Model incorporates result in response

---

## 📝 Examples Included

- `agent_demo.py` - Full agent demonstrations
- `train_conversational.py` - Training script
- `instruction_data.py` - Dataset loaders
- `tools.py` - Tool framework
- `agent.py` - Agent system
- `rlhf.py` - Preference learning

---

## 🏆 Achievement Unlocked

**LILITH is now a conversational AI assistant!**

You can:
- 💬 Have natural conversations
- 🛠️ Use tools autonomously
- 🧠 Complete complex tasks
- 📚 Learn from feedback
- 🎯 Act like Claude (me!)

**Welcome to Phase 3! 🎉**
```

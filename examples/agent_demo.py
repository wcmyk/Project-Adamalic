"""Example: Using LILITH as an autonomous agent.

This demonstrates LILITH's agent capabilities:
- Multi-turn conversations
- Tool use (code execution, calculator)
- Autonomous task completion
"""
from LILITH import load_checkpoint, create_logger
from LILITH.agent import create_assistant_agent, LILITHAgent, AgentConfig
from LILITH.tools import create_default_tools

# Load trained conversational model
print("Loading LILITH conversational model...")
model, tokenizer, _ = load_checkpoint("checkpoints/conversational/conversational_model.pt")
print(f"✓ Model loaded ({model.count_parameters():,} parameters)\n")

# Create assistant agent
print("Creating AI assistant agent...")
agent = create_assistant_agent(model, tokenizer, personality="helpful")
print("✓ Agent ready!\n")

print("=" * 60)
print("DEMO 1: Simple Conversation")
print("=" * 60)

response = agent.chat("Hello! What can you help me with?")
print(f"Agent: {response}\n")

response = agent.chat("What is the capital of France?")
print(f"Agent: {response}\n")

# Reset for next demo
agent.reset()

print("=" * 60)
print("DEMO 2: Tool Use - Calculator")
print("=" * 60)

response = agent.chat("What is 1234 * 5678?", verbose=True)
print(f"Agent: {response}\n")

# Reset
agent.reset()

print("=" * 60)
print("DEMO 3: Tool Use - Code Execution")
print("=" * 60)

response = agent.chat(
    "Write and execute Python code to calculate the sum of numbers from 1 to 100",
    verbose=True
)
print(f"Agent: {response}\n")

# Reset
agent.reset()

print("=" * 60)
print("DEMO 4: Autonomous Task Completion")
print("=" * 60)

task = """Calculate the fibonacci sequence up to 10 terms,
then find the sum of all even numbers in the sequence."""

responses = agent.autonomous_loop(task, max_iterations=5, verbose=True)

print("\n" + "=" * 60)
print("DEMO 5: Multi-turn Conversation")
print("=" * 60)

# Reset
agent.reset()

conversation = [
    "I'm working on a Python project",
    "I need to read a CSV file and process the data",
    "Can you show me how to do that?",
    "What if the CSV has missing values?",
    "Thanks! Can you also show me how to save the results?",
]

for user_msg in conversation:
    print(f"\nYou: {user_msg}")
    response = agent.chat(user_msg)
    print(f"Agent: {response}")

print("\n" + "=" * 60)
print("✓ All demos complete!")
print("=" * 60)

# Save conversation
agent.save_conversation("conversation_log.json")
print("\nConversation saved to conversation_log.json")

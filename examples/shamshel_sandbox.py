"""Example: Using SHAMSHEL sandbox runner for safe code execution."""
from SHAMSHEL.runner_enhanced import EnhancedSandboxRunner

# Create enhanced sandbox runner with security validation
print("Creating enhanced sandbox runner with security features...")
runner = EnhancedSandboxRunner(
    timeout_sec=5.0,
    max_memory_mb=100,
    validate_code=True,
    strict_mode=False,  # Allow more imports for demo
    auto_cleanup=True,
)

# Example 1: Safe code execution
print("\n1. Running safe Python code:")
safe_code = """
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result}")
"""
result = runner.run_python(safe_code)
print(f"   Success: {result.exit_code == 0}")
print(f"   Output: {result.stdout.strip()}")

# Example 2: Code with security violation
print("\n2. Attempting to run code with dangerous import:")
dangerous_code = """
import os
print(os.listdir('/'))
"""
result = runner.run_python(dangerous_code)
if result.security_violations:
    print(f"   Blocked! Reason:")
    print(f"   {result.security_violations}")

# Example 3: Code with syntax error
print("\n3. Running code with syntax error:")
bad_code = """
print('missing closing quote
"""
result = runner.run_python(bad_code)
print(f"   Success: {result.exit_code == 0}")
if result.stderr:
    print(f"   Error: {result.stderr.strip()[:100]}...")

# Example 4: Code with infinite loop (will timeout)
print("\n4. Running code with infinite loop (will timeout):")
infinite_loop = """
while True:
    pass
"""
result = runner.run_python(infinite_loop)
print(f"   Timed out: {result.timed_out}")
print(f"   Duration: {result.duration:.2f}s")

# Example 5: Using context manager for automatic cleanup
print("\n5. Using context manager:")
with EnhancedSandboxRunner(timeout_sec=2) as sandbox:
    code = "print('Hello from sandbox!')"
    result = sandbox.run_python(code)
    print(f"   Output: {result.stdout.strip()}")
print("   Sandbox automatically cleaned up!")

print("\n✓ All examples completed")

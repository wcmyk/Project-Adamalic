"""Tests for SHAMSHEL sandbox runner."""
import pytest
import time
from SHAMSHEL.runner import SandboxRunner, SandboxResult


class TestSandboxRunner:
    """Test suite for SandboxRunner."""

    @pytest.fixture
    def runner(self):
        """Create test sandbox runner."""
        return SandboxRunner(timeout_sec=5, max_memory_mb=100)

    def test_runner_initialization(self, runner):
        """Test runner initializes with correct configuration."""
        assert runner.timeout_sec == 5
        assert runner.max_memory_mb == 100

    def test_run_python_simple(self, runner):
        """Test running simple Python code."""
        code = "print('hello world')"
        result = runner.run_python(code)

        assert isinstance(result, SandboxResult)
        assert result.success
        assert "hello world" in result.stdout
        assert result.stderr == ""
        assert result.exit_code == 0

    def test_run_python_with_output(self, runner):
        """Test capturing Python output."""
        code = """
for i in range(3):
    print(f"Number: {i}")
"""
        result = runner.run_python(code)

        assert result.success
        assert "Number: 0" in result.stdout
        assert "Number: 1" in result.stdout
        assert "Number: 2" in result.stdout

    def test_run_python_with_error(self, runner):
        """Test handling Python errors."""
        code = "raise ValueError('test error')"
        result = runner.run_python(code)

        assert not result.success
        assert "ValueError" in result.stderr
        assert "test error" in result.stderr

    def test_run_python_syntax_error(self, runner):
        """Test handling syntax errors."""
        code = "print('missing closing quote"
        result = runner.run_python(code)

        assert not result.success
        assert "SyntaxError" in result.stderr or "error" in result.stderr.lower()

    def test_run_python_timeout(self):
        """Test timeout enforcement."""
        runner = SandboxRunner(timeout_sec=1, max_memory_mb=100)
        code = """
import time
time.sleep(5)
print('should not reach here')
"""
        result = runner.run_python(code)

        assert not result.success
        assert result.timed_out

    def test_run_shell_simple(self, runner):
        """Test running simple shell commands."""
        script = "echo 'test'"
        result = runner.run_shell(script)

        assert result.success
        assert "test" in result.stdout

    def test_run_shell_with_exit_code(self, runner):
        """Test shell exit codes."""
        script = "exit 1"
        result = runner.run_shell(script)

        assert not result.success
        assert result.exit_code == 1

    def test_run_shell_timeout(self):
        """Test shell timeout."""
        runner = SandboxRunner(timeout_sec=1, max_memory_mb=100)
        script = "sleep 5"
        result = runner.run_shell(script)

        assert not result.success
        assert result.timed_out

    def test_resource_limits_applied(self, runner):
        """Test that resource limits are configured."""
        # This is a basic test - actual resource limiting requires
        # running the sandbox and checking if limits are enforced
        assert runner.timeout_sec > 0
        assert runner.max_memory_mb > 0

    def test_multiple_runs(self, runner):
        """Test running multiple isolated executions."""
        code1 = "x = 42\nprint(x)"
        code2 = "print(x)"  # x should not be defined

        result1 = runner.run_python(code1)
        assert result1.success
        assert "42" in result1.stdout

        result2 = runner.run_python(code2)
        assert not result2.success  # NameError: x not defined

    @pytest.mark.parametrize("code,expected", [
        ("print(1 + 1)", "2"),
        ("print('hello' * 3)", "hellohellohello"),
        ("import math\nprint(math.pi)", "3.14"),
    ])
    def test_various_python_operations(self, runner, code, expected):
        """Test various Python operations."""
        result = runner.run_python(code)
        assert result.success
        assert expected in result.stdout

"""Enhanced sandbox runner with security validation and resource cleanup."""
from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .runner import SandboxRunner as BaseSandboxRunner, SandboxSpec, SandboxResult
from .security import sanitize_python_code


@dataclass
class EnhancedSandboxResult(SandboxResult):
    """Enhanced result with security information."""

    security_violations: Optional[str] = None
    was_validated: bool = False


class EnhancedSandboxRunner:
    """Enhanced sandbox runner with security validation and automatic cleanup.

    Features:
    - AST validation before code execution
    - Automatic cleanup of temporary directories
    - Stricter security controls
    - Resource limit enforcement
    """

    def __init__(
        self,
        timeout_sec: float = 5.0,
        max_memory_mb: int = 256,
        validate_code: bool = True,
        strict_mode: bool = True,
        auto_cleanup: bool = True,
    ):
        """Initialize enhanced sandbox runner.

        Args:
            timeout_sec: Execution timeout in seconds
            max_memory_mb: Memory limit in MB
            validate_code: Enable AST validation
            strict_mode: Use strict validation (whitelist approach)
            auto_cleanup: Automatically cleanup temp directories
        """
        self.timeout_sec = timeout_sec
        self.max_memory_mb = max_memory_mb
        self.validate_code = validate_code
        self.strict_mode = strict_mode
        self.auto_cleanup = auto_cleanup

        # Create temp directory for sandboxes
        self.temp_dir = Path(tempfile.mkdtemp(prefix="shamshel-"))
        self.base_runner = BaseSandboxRunner(base_dir=self.temp_dir)

        # Track created directories for cleanup
        self.created_dirs: set[Path] = set()
        self.created_dirs.add(self.temp_dir)

        # Register cleanup on exit
        if auto_cleanup:
            atexit.register(self.cleanup_all)

    def run_python(self, code: str) -> EnhancedSandboxResult:
        """Run Python code in sandbox with security validation.

        Args:
            code: Python code to execute

        Returns:
            EnhancedSandboxResult with execution details
        """
        # Validate code if enabled
        security_violations = None
        if self.validate_code:
            is_safe, error_msg = sanitize_python_code(code, strict_mode=self.strict_mode)
            if not is_safe:
                return EnhancedSandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration=0.0,
                    timed_out=False,
                    workdir=self.temp_dir,
                    security_violations=error_msg,
                    was_validated=True,
                )

        # Create sandbox spec
        spec = SandboxSpec(
            timeout=self.timeout_sec,
            cpu_time_limit=int(self.timeout_sec),
            memory_limit_mb=self.max_memory_mb,
        )

        # Execute code
        result = self.base_runner.run_python(code, spec)

        # Track workdir for cleanup
        if result.workdir not in self.created_dirs:
            self.created_dirs.add(result.workdir)

        # Return enhanced result
        return EnhancedSandboxResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=result.duration,
            timed_out=result.timed_out,
            workdir=result.workdir,
            security_violations=security_violations,
            was_validated=self.validate_code,
        )

    def run_shell(self, script: str) -> EnhancedSandboxResult:
        """Run shell script in sandbox.

        Args:
            script: Shell script to execute

        Returns:
            EnhancedSandboxResult with execution details
        """
        # Shell scripts are not validated (could be added)
        spec = SandboxSpec(
            timeout=self.timeout_sec,
            cpu_time_limit=int(self.timeout_sec),
            memory_limit_mb=self.max_memory_mb,
        )

        result = self.base_runner.run_shell(script, spec)

        # Track workdir for cleanup
        if result.workdir not in self.created_dirs:
            self.created_dirs.add(result.workdir)

        return EnhancedSandboxResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=result.duration,
            timed_out=result.timed_out,
            workdir=result.workdir,
            security_violations=None,
            was_validated=False,
        )

    def cleanup_workdir(self, workdir: Path):
        """Clean up a specific working directory.

        Args:
            workdir: Directory to clean up
        """
        if workdir.exists() and workdir in self.created_dirs:
            try:
                shutil.rmtree(workdir)
                self.created_dirs.discard(workdir)
            except Exception:
                pass  # Best effort cleanup

    def cleanup_all(self):
        """Clean up all created directories."""
        for workdir in list(self.created_dirs):
            if workdir.exists():
                try:
                    shutil.rmtree(workdir)
                except Exception:
                    pass  # Best effort cleanup
        self.created_dirs.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.auto_cleanup:
            self.cleanup_all()

    def __del__(self):
        """Cleanup on deletion."""
        if self.auto_cleanup:
            self.cleanup_all()


# Convenience function maintaining backward compatibility
class SandboxRunner(EnhancedSandboxRunner):
    """Alias for backward compatibility."""

    def __init__(self, timeout_sec: float = 5.0, max_memory_mb: int = 256):
        super().__init__(
            timeout_sec=timeout_sec,
            max_memory_mb=max_memory_mb,
            validate_code=True,
            strict_mode=False,  # More permissive by default for compatibility
            auto_cleanup=True,
        )


__all__ = [
    "EnhancedSandboxRunner",
    "EnhancedSandboxResult",
    "SandboxRunner",  # Backward compatibility
]

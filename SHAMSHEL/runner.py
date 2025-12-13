"""Local sandbox runner that executes short-lived commands with resource limits."""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class SandboxSpec:
    """Execution constraints for a sandboxed subprocess."""

    timeout: float = 5.0
    cpu_time_limit: int = 2  # seconds of CPU time
    memory_limit_mb: int = 256
    env: Optional[Dict[str, str]] = None
    stdin: Optional[str] = None
    workdir: Optional[Path] = None
    command_prefix: List[str] = field(default_factory=list)


@dataclass
class SandboxResult:
    """Outcome of a sandboxed execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool
    workdir: Path


class SandboxRunner:
    """Create short-lived sandboxes with resource limits for running generated code."""

    def __init__(self, base_dir: Path | str = "artifacts/shamshel"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run_command(self, command: Iterable[str], spec: SandboxSpec) -> SandboxResult:
        sandbox_dir = spec.workdir or Path(tempfile.mkdtemp(prefix="sandbox-", dir=self.base_dir))
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        env = {"PATH": "/usr/bin:/bin"}
        if spec.env:
            env.update(spec.env)
        full_command: List[str] = list(spec.command_prefix) + list(command)
        start = time.monotonic()
        try:
            completed = subprocess.run(
                full_command,
                input=spec.stdin,
                capture_output=True,
                text=True,
                timeout=spec.timeout,
                cwd=sandbox_dir,
                env=env,
                preexec_fn=lambda: _apply_limits(spec),
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = exc
            timed_out = True
        duration = time.monotonic() - start
        stdout = completed.stdout if hasattr(completed, "stdout") else ""
        stderr = completed.stderr if hasattr(completed, "stderr") else ""
        exit_code = completed.returncode if hasattr(completed, "returncode") else -1
        return SandboxResult(
            exit_code=exit_code if not timed_out else -1,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            timed_out=timed_out,
            workdir=sandbox_dir,
        )

    def run_python(self, code: str, spec: SandboxSpec) -> SandboxResult:
        sandbox_dir = spec.workdir or Path(tempfile.mkdtemp(prefix="py-sandbox-", dir=self.base_dir))
        script_path = sandbox_dir / "main.py"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(code)
        return self.run_command([sys.executable, str(script_path)], spec)

    def run_shell(self, script: str, spec: SandboxSpec) -> SandboxResult:
        sandbox_dir = spec.workdir or Path(tempfile.mkdtemp(prefix="sh-sandbox-", dir=self.base_dir))
        script_path = sandbox_dir / "script.sh"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        return self.run_command(["bash", str(script_path)], spec)


def _apply_limits(spec: SandboxSpec) -> None:
    if spec.cpu_time_limit:
        resource.setrlimit(resource.RLIMIT_CPU, (spec.cpu_time_limit, spec.cpu_time_limit))
    if spec.memory_limit_mb:
        byte_limit = spec.memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (byte_limit, byte_limit))
    # Drop supplemental groups and isolate environment as best as possible without containers
    os.setsid()

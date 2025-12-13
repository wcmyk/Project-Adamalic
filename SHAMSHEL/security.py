"""Security utilities for code validation and sanitization."""
from __future__ import annotations

import ast
from typing import List, Set, Tuple


class CodeSecurityValidator:
    """Validates code for security issues before execution."""

    # Dangerous imports that should be blocked
    DANGEROUS_IMPORTS = {
        "os", "subprocess", "sys", "shutil", "socket", "urllib",
        "requests", "http", "ftplib", "smtplib", "telnetlib",
        "pickle", "shelve", "marshal", "eval", "exec", "compile",
        "__import__", "importlib", "ctypes", "multiprocessing",
    }

    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {
        "eval", "exec", "compile", "__import__", "open",
        "input", "raw_input", "execfile",
    }

    # Allowed imports (whitelist approach for safer execution)
    ALLOWED_IMPORTS = {
        "math", "random", "itertools", "functools", "collections",
        "datetime", "json", "re", "string", "textwrap",
    }

    def __init__(self, strict_mode: bool = True):
        """Initialize validator.

        Args:
            strict_mode: If True, only allow whitelisted imports
        """
        self.strict_mode = strict_mode
        self.violations: List[str] = []

    def validate_python_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python code for security issues.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        self.violations = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.violations.append(f"Syntax error: {e}")
            return False, self.violations

        # Check for dangerous patterns
        self._check_imports(tree)
        self._check_dangerous_calls(tree)
        self._check_file_operations(tree)
        self._check_infinite_loops(tree)
        self._check_recursion_depth(tree)

        return len(self.violations) == 0, self.violations

    def _check_imports(self, tree: ast.AST):
        """Check for dangerous imports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if self.strict_mode and module not in self.ALLOWED_IMPORTS:
                        self.violations.append(
                            f"Import '{module}' not in allowed list (strict mode)"
                        )
                    elif module in self.DANGEROUS_IMPORTS:
                        self.violations.append(
                            f"Dangerous import detected: '{module}'"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if self.strict_mode and module not in self.ALLOWED_IMPORTS:
                        self.violations.append(
                            f"Import from '{module}' not in allowed list (strict mode)"
                        )
                    elif module in self.DANGEROUS_IMPORTS:
                        self.violations.append(
                            f"Dangerous import from: '{module}'"
                        )

    def _check_dangerous_calls(self, tree: ast.AST):
        """Check for dangerous function calls."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None

                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in self.DANGEROUS_BUILTINS:
                    self.violations.append(
                        f"Dangerous function call: '{func_name}'"
                    )

    def _check_file_operations(self, tree: ast.AST):
        """Check for file operations."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    self.violations.append(
                        "File operations not allowed: 'open()'"
                    )

            # Check for with open(...) statements
            if isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name):
                            if item.context_expr.func.id == "open":
                                self.violations.append(
                                    "File operations not allowed: 'with open()'"
                                )

    def _check_infinite_loops(self, tree: ast.AST):
        """Check for potential infinite loops."""
        for node in ast.walk(tree):
            # while True without break is suspicious
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    has_break = self._has_break_or_return(node)
                    if not has_break:
                        self.violations.append(
                            "Potential infinite loop detected: 'while True' without break/return"
                        )

    def _has_break_or_return(self, node: ast.AST) -> bool:
        """Check if a node contains break or return statements."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Break, ast.Return)):
                return True
        return False

    def _check_recursion_depth(self, tree: ast.AST):
        """Check for deeply nested structures that might cause recursion issues."""
        def get_depth(node: ast.AST, current_depth: int = 0) -> int:
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                child_depth = get_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            return max_depth

        depth = get_depth(tree)
        if depth > 50:  # Arbitrary limit
            self.violations.append(
                f"Code structure too deeply nested (depth: {depth})"
            )


def sanitize_python_code(code: str, strict_mode: bool = True) -> Tuple[bool, str]:
    """Sanitize and validate Python code.

    Args:
        code: Python code to sanitize
        strict_mode: Enable strict validation

    Returns:
        Tuple of (is_safe, error_message_or_empty)
    """
    validator = CodeSecurityValidator(strict_mode=strict_mode)
    is_safe, violations = validator.validate_python_code(code)

    if not is_safe:
        error_msg = "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations)
        return False, error_msg

    return True, ""


def get_safe_builtins() -> dict:
    """Get a dictionary of safe built-in functions for exec/eval.

    Returns:
        Dictionary of safe built-ins
    """
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }
    return safe_builtins


__all__ = [
    "CodeSecurityValidator",
    "sanitize_python_code",
    "get_safe_builtins",
]

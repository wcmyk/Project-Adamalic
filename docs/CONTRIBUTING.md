# Contributing to Project Adamalic (AngelOS)

Thank you for your interest in contributing to Project Adamalic! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow best practices for collaboration

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Project-Adamalic.git
   cd Project-Adamalic
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/wcmyk/Project-Adamalic.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv
- Git

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install in editable mode with dev dependencies
   ```

3. **Verify installation**:
   ```bash
   pytest tests/
   ```

## Project Structure

```
Project-Adamalic/
├── LILITH/              # Dual-role LLM system
│   ├── config.py       # Configuration classes
│   ├── data.py         # Data utilities and tokenizers
│   ├── model.py        # GPT decoder implementation
│   ├── system.py       # Multi-LLM coordinator
│   ├── train.py        # Training loops
│   ├── sampling.py     # Advanced sampling strategies
│   ├── lora.py         # LoRA fine-tuning
│   └── ...
├── SHAMSHEL/            # Sandbox runner
│   ├── runner.py       # Basic sandbox runner
│   ├── runner_enhanced.py  # Enhanced with security
│   └── security.py     # Security validation
├── tests/               # Test suite
│   ├── test_lilith/
│   └── test_shamshel/
├── examples/            # Example scripts
└── ...
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-attention-visualization`
- `fix/memory-leak-in-cache`
- `docs/improve-training-guide`

### Commit Messages

Follow conventional commit format:
```
type(scope): brief description

Longer explanation if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(lilith): add beam search decoding

Implements beam search with configurable beam width
and length penalty for improved generation quality.

Closes #42
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_lilith/test_model.py

# Run with coverage
pytest --cov=LILITH --cov=SHAMSHEL --cov-report=html

# Run specific test
pytest tests/test_lilith/test_model.py::TestGPTDecoder::test_generation
```

### Writing Tests

- Place tests in the appropriate `tests/test_*/` directory
- Use descriptive test names: `test_<functionality>_<scenario>`
- Include docstrings explaining what the test validates
- Use fixtures for reusable test components
- Aim for high code coverage (>80%)

Example:
```python
def test_model_generates_correct_length():
    """Test that generation produces expected sequence length."""
    model = GPTDecoder(config)
    prompt = torch.randint(0, 100, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    assert generated.shape == (1, 30)  # 10 + 20
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 80)
- **Imports**: Use absolute imports, group by stdlib/third-party/local
- **Type hints**: Required for all function signatures
- **Docstrings**: Google style for all public APIs

### Type Hints

```python
def train(
    corpus: Iterable[str],
    model_config: ModelConfig,
    train_config: TrainingConfig,
) -> GPTDecoder:
    """Train a model on a corpus.

    Args:
        corpus: Training texts
        model_config: Model configuration
        train_config: Training configuration

    Returns:
        Trained model
    """
    pass
```

### Docstrings

Use Google style:

```python
def calculate_perplexity(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculate perplexity on a dataset.

    Perplexity is exp(average_loss), a standard metric for language models.
    Lower is better.

    Args:
        model: The language model
        dataloader: DataLoader for evaluation data

    Returns:
        Perplexity score

    Raises:
        ValueError: If dataloader is empty
    """
    pass
```

### Code Formatting

Use these tools:

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy LILITH/ SHAMSHEL/

# Linting
flake8 LILITH/ SHAMSHEL/
```

## Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin your-branch-name
   ```

3. **Create pull request** on GitHub

4. **PR Checklist**:
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated (if needed)
   - [ ] CHANGELOG.md updated (for notable changes)
   - [ ] Type hints added
   - [ ] Docstrings added/updated

5. **PR Description** should include:
   - What changes were made
   - Why the changes were needed
   - How to test the changes
   - Related issues (Fixes #123)

### Review Process

- Maintainers will review your PR
- Address feedback promptly
- Keep discussions professional and constructive
- Be patient - reviews take time!

## Reporting Issues

### Bug Reports

Include:
- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Minimal reproducible example

Template:
```markdown
**Environment:**
- Python version: 3.10.5
- OS: Ubuntu 22.04
- PyTorch version: 2.0.1

**Description:**
Model training crashes with OOM error...

**Steps to Reproduce:**
1. Create model with config X
2. Run training with batch size Y
3. See error

**Expected:** Training should complete
**Actual:** OOM error

**Error Message:**
```
[paste error here]
```

**Minimal Example:**
```python
[paste code here]
```
```

### Feature Requests

Include:
- Clear description of the feature
- Use cases and benefits
- Proposed implementation (if you have ideas)
- Willingness to contribute

## Development Guidelines

### Performance

- Profile before optimizing
- Use appropriate data structures
- Consider memory usage
- Document performance characteristics

### Security

- Validate all user inputs
- Never execute arbitrary code without sandboxing
- Follow security best practices
- Report security issues privately

### Documentation

- Keep README.md up to date
- Document all public APIs
- Include examples for new features
- Update docstrings when changing behavior

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email maintainers privately
- **General**: Check existing issues and documentation

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Git commit history

Thank you for contributing to Project Adamalic! 🚀

# Contributing to HopfieldSPP

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/HopfieldSPP.git
   cd HopfieldSPP
   ```
3. Set up the development environment:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # if applicable
   ```
4. Verify TensorFlow is installed correctly:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature` or `git checkout -b fix/issue-description`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Commit with a clear message: `git commit -m "feat: add new feature"`
5. Push and open a Pull Request

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

All new code should include tests. Aim to maintain or improve coverage.

## Project Structure

- **Models:** Three versions exist (Original, Improved, Advanced). The Improved model is recommended for most work.
- **Training:** `python src/train_model_improved.py`
- **API:** `python src/main_improved.py` (FastAPI with caching)
- **Benchmarks:** `python benchmark_all.py`

## Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Add docstrings to public functions and classes
- Keep TensorFlow operations vectorized where possible
- Use clear, descriptive variable names

## Commit Messages

Use [conventional commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `test:` adding or updating tests
- `refactor:` code restructuring

## Reporting Issues

- Use the issue templates (Bug Report or Feature Request)
- Include steps to reproduce for bugs
- Mention TensorFlow version and hardware (CPU/GPU) when relevant
- Check existing issues before creating a new one

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to learn and build.

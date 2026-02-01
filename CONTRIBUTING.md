# Contributing to Financial Forecasting Application

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and beginners
- Focus on constructive feedback
- Help create a positive environment

## How Can You Contribute?

### 1. Reporting Bugs

Found a bug? Please create an issue with:

- **Clear title** describing the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details**: OS, Python version
- **Error messages** or screenshots
- **Data details**: CSV structure (if relevant)

**Example:**
```
Title: LSTM model fails with 1-day horizon

Steps:
1. Load AAPL data
2. Select 1-day horizon
3. Click generate forecast

Expected: Forecast completes successfully
Actual: Error "ValueError: not enough values to unpack"

Environment: Windows 10, Python 3.10.5
```

### 2. Suggesting Enhancements

Have an idea? Create an issue with:

- **Feature description**: What you want added
- **Use case**: Why it's useful
- **Implementation ideas**: How it might work
- **Examples**: Similar features elsewhere

### 3. Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or use cases
- Improve installation instructions
- Translate to other languages
- Add diagrams or screenshots

### 4. Contributing Code

#### Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/financial-forecasting.git
   cd financial-forecasting
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # if exists
   ```

#### Development Workflow

1. **Make your changes**
2. **Follow code style** (see below)
3. **Add tests** for new features
4. **Run tests**:
   ```bash
   pytest tests/ -v
   ```
5. **Update documentation** as needed
6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create Pull Request** on GitHub

## Code Style Guidelines

### Python Style

Follow PEP 8 with these specifics:

```python
# Good: Clear function names, docstrings, type hints
def calculate_forecast(
    data: pd.DataFrame, 
    horizon: int = 7
) -> np.ndarray:
    """
    Calculate forecast for given horizon.
    
    Args:
        data: Historical price data
        horizon: Number of days to forecast
        
    Returns:
        Array of forecasted values
    """
    # Implementation
    pass

# Bad: No docstring, unclear names
def calc(d, h=7):
    pass
```

### Formatting

Use automatic formatters:

```bash
# Format with black
black forecasting_app/

# Check style with flake8
flake8 forecasting_app/

# Type checking with mypy (optional)
mypy forecasting_app/
```

### Code Organization

```python
# 1. Standard library imports
import os
import sys
from datetime import datetime

# 2. Third-party imports
import pandas as pd
import numpy as np
from flask import Flask, render_template

# 3. Local imports
from forecasting_app.models import arima_forecast
from forecasting_app.utils import load_data
```

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

```python
# Good
class ForecastModel:
    MAX_HORIZON = 14
    
    def __init__(self):
        self.data = None
    
    def _preprocess_data(self, df):
        pass
    
    def generate_forecast(self, horizon):
        pass
```

## Testing Guidelines

### Writing Tests

Add tests for all new features:

```python
import pytest
from forecasting_app.models import arima_forecast

def test_arima_forecast_returns_correct_length():
    """ARIMA forecast should return array of length horizon."""
    df = create_sample_data()  # helper function
    horizon = 7
    
    result = arima_forecast(df, horizon)
    
    assert len(result) == horizon
    assert all(isinstance(x, float) for x in result)

def test_arima_forecast_handles_invalid_input():
    """ARIMA forecast should raise ValueError for invalid input."""
    df = pd.DataFrame()  # empty dataframe
    
    with pytest.raises(ValueError):
        arima_forecast(df, horizon=7)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=forecasting_app --cov-report=html

# Run specific test
pytest tests/test_models.py::test_arima_forecast -v
```

### Test Coverage

Aim for:
- **Minimum**: 70% coverage
- **Target**: 85% coverage
- **New features**: 90%+ coverage

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No linting errors
- [ ] Commits are logical and well-described
- [ ] Branch is up to date with main

### PR Title Format

Use conventional commits:

- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Improve existing feature
- `Docs:` Documentation changes
- `Test:` Test additions/changes
- `Refactor:` Code restructuring
- `Style:` Formatting changes

**Examples:**
- `Add: Support for additional stock symbols`
- `Fix: LSTM model memory leak issue`
- `Update: Improve chart rendering performance`
- `Docs: Add examples to user guide`

### PR Description

Include:

1. **What**: Brief description of changes
2. **Why**: Reason for changes
3. **How**: Implementation approach
4. **Testing**: How you tested
5. **Screenshots**: If UI changes
6. **Breaking Changes**: If any
7. **Related Issues**: Link to issues

**Template:**
```markdown
## Description
Brief description of what this PR does.

## Motivation
Why this change is needed.

## Changes Made
- Added feature X
- Fixed bug Y
- Updated documentation Z

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] All tests passing

## Screenshots
(if applicable)

## Breaking Changes
None / List any breaking changes

## Related Issues
Closes #123
```

### Review Process

1. **Automated checks** run (tests, linting)
2. **Maintainer review** (1-2 business days)
3. **Feedback addressed** (if needed)
4. **Approval and merge**

## Development Setup

### Recommended Tools

- **IDE**: VS Code, PyCharm
- **Extensions**: 
  - Python
  - Pylance
  - Black Formatter
  - GitLens
- **Tools**:
  - Git
  - Docker (optional)
  - Postman (for API testing)

### Environment Variables

Create `.env` file:

```bash
FLASK_ENV=development
FLASK_DEBUG=True
DATABASE_URL=sqlite:///forecasting.db
SECRET_KEY=dev-secret-key
```

### Database Migrations

If changing database schema:

```bash
# Create migration
alembic revision --autogenerate -m "Add new column"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Areas Needing Help

Current priorities:

### High Priority
- [ ] Additional stock symbols support
- [ ] Real-time data integration
- [ ] Model performance optimization
- [ ] Mobile responsiveness improvements

### Medium Priority
- [ ] Additional forecasting models (Prophet, GRU)
- [ ] Backtesting framework
- [ ] User authentication
- [ ] API endpoint documentation

### Low Priority
- [ ] Dark mode theme
- [ ] Export to PDF
- [ ] Email notifications
- [ ] Social media integration

## Questions?

- **General**: Open a discussion on GitHub
- **Bugs**: Create an issue
- **Security**: Email maintainers directly
- **Chat**: Join our Discord/Slack (if available)

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🎉

Every contribution, no matter how small, makes a difference.


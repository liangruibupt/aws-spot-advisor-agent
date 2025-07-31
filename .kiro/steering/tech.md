# Technology Stack

## Core Technologies
- **Language**: Python 3.8+
- **AWS Integration**: AWS Bedrock AgentCore, boto3
- **Data Processing**: Pydantic for validation, dataclasses for models
- **Web Scraping**: AWS Bedrock AgentCore (not traditional scraping libraries)
- **Testing**: pytest, moto for AWS mocking
- **Utilities**: python-dotenv, tenacity for retries, functools for caching

## Project Structure
- Use dataclasses for data models with type hints
- Implement service-oriented architecture with clear separation of concerns
- Follow dependency injection patterns for testability
- Use Pydantic for input validation where needed

## Common Commands

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### Testing
```bash
pytest                    # Run all tests
pytest -v                # Verbose output
pytest tests/unit/       # Unit tests only
pytest tests/integration/ # Integration tests only
```

### Development
```bash
python -m pip install -e .  # Install in development mode
python -m mypy src/         # Type checking
python -m black src/        # Code formatting
python -m flake8 src/       # Linting
```

## Code Style Guidelines
- Use type hints for all function parameters and return values
- Follow PEP 8 naming conventions
- Add comprehensive docstrings for all public methods
- Implement proper error handling with custom exception classes
- Use logging instead of print statements
- Keep functions focused and testable (single responsibility)
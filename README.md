# AWS Spot Price Analyzer

A Python service that analyzes AWS EC2 spot pricing data to identify the most cost-effective regions for high-performance computing instances (p5en.48xlarge and p5.48xlarge).

## Features

- Real-time spot price analysis using AWS Bedrock AgentCore for web scraping
- Intelligent filtering based on interruption rates (<5%)
- Automated ranking and recommendation of top 3 regions
- JSON API responses with structured data
- Built-in caching and data freshness validation

## Installation

### Prerequisites

- Python 3.8 or higher
- AWS credentials configured
- Virtual environment (recommended)

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install in development mode:
```bash
pip install -e .
```

## Usage

```python
from src.services.spot_price_analyzer import SpotPriceAnalyzer

analyzer = SpotPriceAnalyzer()
results = analyzer.analyze_spot_prices(['p5en.48xlarge', 'p5.48xlarge'])
print(results)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Configuration

Set the following environment variables:

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_DEFAULT_REGION`: Default AWS region
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## License

MIT License
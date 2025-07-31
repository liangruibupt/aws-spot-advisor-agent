# Project Structure

## Directory Organization

```
aws-spot-advisor-agent/
├── src/
│   ├── models/           # Data models and dataclasses
│   ├── services/         # Business logic services
│   ├── utils/           # Utility functions and helpers
│   └── main.py          # Application entry point
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── fixtures/        # Test data and fixtures
├── config/              # Configuration files
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
└── README.md
```

## Key Components

### Services Layer
- `SpotPriceAnalyzer`: Main orchestration service
- `WebScraperService`: Handles AWS Bedrock AgentCore web scraping
- `DataFilterService`: Filters data by interruption rates
- `RankingEngine`: Sorts and ranks regions by price
- `BedrockAgentService`: AWS Bedrock AgentCore wrapper

### Models Layer
- `RawSpotData`: Raw scraped data structure
- `SpotPriceResult`: Processed result data
- `AnalysisResponse`: Final API response format

### Utils Layer
- Configuration management
- Logging setup
- Custom exceptions
- Caching utilities

## Naming Conventions
- Use descriptive class names ending with purpose (Service, Engine, etc.)
- Method names should be verbs describing the action
- Constants in UPPER_CASE
- Private methods prefixed with underscore
- Test files mirror source structure with `test_` prefix

## Import Organization
1. Standard library imports
2. Third-party imports
3. Local application imports
4. Separate each group with blank line
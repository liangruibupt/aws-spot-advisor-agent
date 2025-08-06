# AWS Spot Price Analyzer - User Guide

## Overview

The AWS Spot Price Analyzer helps you find the most cost-effective AWS regions for running high-performance GPU instances (p5en.48xlarge and p5.48xlarge) with low interruption rates. It analyzes real-time spot pricing data and recommends the top 3 regions based on price and reliability.

## Quick Start

### 1. Installation & Setup

```bash
# Clone the repository
git clone <repository-url>
cd aws-spot-advisor-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure AWS Credentials

Set up your AWS credentials using one of these methods:

**Option A: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

**Option B: Create .env file**
```bash
# Create .env file in project root
cat > .env << EOF
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
LOG_LEVEL=INFO
EOF
```

**Option C: AWS CLI Configuration**
```bash
aws configure
```

### 3. Basic Usage

**Command Line (Recommended)**

```bash
# Run as a module from project root (recommended approach)
python -m src.main

# Alternative: Using the launcher script
python run_analyzer.py

# Install as package and use console script
pip install -e .
aws-spot-analyzer
```

**Try the Demo First**

Before setting up AWS credentials, you can run the demo to see how the system works:

```bash
# Run the interactive demo (works without AWS credentials)
python demo_usage.py
```

This demo shows:
- How to initialize the analyzer
- Mock data processing workflow
- JSON output format
- Configuration options

**Python Script**
```python
from src.services.spot_price_analyzer import SpotPriceAnalyzer

# Initialize analyzer
analyzer = SpotPriceAnalyzer()

# Get analysis results as JSON
results = analyzer.analyze_spot_prices_json()
print(results)
```

## Usage Examples

### Example 1: Basic Analysis
```python
from src.services.spot_price_analyzer import SpotPriceAnalyzer

analyzer = SpotPriceAnalyzer()

# Analyze with default settings
results = analyzer.analyze_spot_prices_json(include_summary=True)

# Pretty print results
import json
print(json.dumps(results, indent=2))
```

**Sample Output:**
```json
{
  "results": [
    {
      "region": "us-west-2",
      "instance_type": "p5en.48xlarge",
      "spot_price": 8.45,
      "currency": "USD",
      "interruption_rate": 2.1,
      "availability": true,
      "rank": 1
    },
    {
      "region": "eu-west-1",
      "instance_type": "p5en.48xlarge", 
      "spot_price": 9.12,
      "currency": "USD",
      "interruption_rate": 1.8,
      "availability": true,
      "rank": 2
    },
    {
      "region": "ap-southeast-1",
      "instance_type": "p5.48xlarge",
      "spot_price": 9.67,
      "currency": "USD", 
      "interruption_rate": 3.2,
      "availability": true,
      "rank": 3
    }
  ],
  "summary": {
    "total_regions_analyzed": 15,
    "regions_meeting_criteria": 8,
    "lowest_price": 8.45,
    "highest_interruption_rate": 3.2,
    "analysis_timestamp": "2025-01-06T10:30:45Z"
  }
}
```

### Example 2: Custom Parameters
```python
from src.services.spot_price_analyzer import SpotPriceAnalyzer

analyzer = SpotPriceAnalyzer()

# Analyze specific instance type with custom criteria
results = analyzer.analyze_spot_prices_json(
    instance_types=["p5en.48xlarge"],  # Only analyze this instance type
    max_interruption_rate=0.03,        # Max 3% interruption rate
    top_count=5,                       # Get top 5 regions instead of 3
    force_refresh=True,                # Force fresh data (bypass cache)
    include_summary=True               # Include analysis summary
)

print(json.dumps(results, indent=2))
```

### Example 3: Error Handling
```python
from src.services.spot_price_analyzer import SpotPriceAnalyzer, SpotPriceAnalyzerError

try:
    analyzer = SpotPriceAnalyzer()
    results = analyzer.analyze_spot_prices_json()
    
    if "error" in results:
        print(f"Analysis failed: {results['error']['message']}")
    else:
        print(f"Found {len(results['results'])} regions")
        
except SpotPriceAnalyzerError as e:
    print(f"Analyzer error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Example 4: Using the Main Functions
```python
from src.main import run_analysis_with_params, run_analysis_json_string

# Get results as dictionary
results_dict = run_analysis_with_params(
    instance_types=["p5.48xlarge"],
    max_interruption_rate=0.04,
    top_count=3,
    include_summary=True
)

# Get results as formatted JSON string
results_json = run_analysis_json_string(
    instance_types=["p5en.48xlarge", "p5.48xlarge"],
    max_interruption_rate=0.05,
    top_count=3,
    indent=4
)

print(results_json)
```

## Configuration Options

You can customize the analyzer behavior using environment variables:

```bash
# Cache settings
export CACHE_TTL_HOURS=2              # Cache data for 2 hours (default: 1)

# Analysis settings  
export MAX_INTERRUPTION_RATE=3.0      # Max interruption rate % (default: 5.0)
export TOP_REGIONS_COUNT=5            # Number of top regions (default: 3)

# Logging
export LOG_LEVEL=DEBUG                # DEBUG, INFO, WARNING, ERROR (default: INFO)

# AWS settings
export AWS_DEFAULT_REGION=eu-west-1   # Default AWS region (default: us-east-1)

# Bedrock model settings
export BEDROCK_MODEL_ID=us.anthropic.claude-3-7-sonnet-20250219-v1:0  # Claude model to use
export BEDROCK_REGION=us-east-1       # Bedrock region (default: us-east-1)
```

### Available Claude Models

The system supports the latest Claude models:

- **Claude 4 Sonnet** (recommended): `us.anthropic.claude-sonnet-4-20250514-v1:0`
- **Claude 3.7 Sonnet**: `us.anthropic.claude-3-7-sonnet-20250219-v1:0`

**Note**: The `us.` prefix is required for cross-region inference. Make sure you have access to these models in your AWS Bedrock account.

## Advanced Usage

### Cache Management
```python
from src.services.spot_price_analyzer import SpotPriceAnalyzer

analyzer = SpotPriceAnalyzer()

# Check cache status
cache_info = analyzer.get_analysis_cache_info()
print(f"Cache size: {cache_info['size']}")
print(f"Hit rate: {cache_info['hit_rate']:.2%}")

# Clear cache
analyzer.clear_cache()

# Warm cache for better performance
analyzer.warm_analysis_cache(
    instance_type_combinations=[["p5en.48xlarge"], ["p5.48xlarge"]],
    max_interruption_rates=[0.03, 0.05],
    top_counts=[3, 5]
)
```

### Getting Detailed Information
```python
analyzer = SpotPriceAnalyzer()

# Get system status and diagnostics
status = analyzer.get_system_status()
print(json.dumps(status, indent=2))

# This includes:
# - Last scrape time
# - Cache statistics  
# - Filter statistics
# - Configuration details
```

## Command Line Usage

### Basic Command
```bash
# Run with defaults (recommended)
python -m src.main

# Alternative: Using launcher script
python run_analyzer.py

# With custom log level
LOG_LEVEL=DEBUG python -m src.main

# Force refresh data
FORCE_REFRESH=true python -m src.main
```

### Using as a Module
```bash
# Run as Python module from project root (standard approach)
python -m src.main

# With environment variables
MAX_INTERRUPTION_RATE=3.0 TOP_REGIONS_COUNT=5 python -m src.main

# Alternative using launcher script
MAX_INTERRUPTION_RATE=3.0 TOP_REGIONS_COUNT=5 python run_analyzer.py
```

## Integration Examples

### Web API Integration
```python
from flask import Flask, jsonify
from src.main import run_analysis_with_params

app = Flask(__name__)

@app.route('/spot-analysis')
def get_spot_analysis():
    results = run_analysis_with_params(include_summary=True)
    return jsonify(results)

@app.route('/spot-analysis/<instance_type>')
def get_spot_analysis_for_instance(instance_type):
    results = run_analysis_with_params(
        instance_types=[instance_type],
        include_summary=True
    )
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
```

### Scheduled Analysis
```python
import schedule
import time
from src.main import run_analysis_json_string

def daily_analysis():
    """Run daily spot price analysis."""
    results = run_analysis_json_string(
        force_refresh=True,
        include_summary=True
    )
    
    # Save to file or send to monitoring system
    with open(f'spot_analysis_{time.strftime("%Y%m%d")}.json', 'w') as f:
        f.write(results)
    
    print(f"Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Schedule daily analysis at 9 AM
schedule.every().day.at("09:00").do(daily_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Common Issues

**1. AWS Credentials Error**
```
Error: Unable to locate credentials
```
**Solution:** Set up AWS credentials using one of the methods in the setup section.

**2. Bedrock Model Access Error**
```
Error: The provided model identifier is invalid
```
**Solution:** Ensure your AWS account has access to AWS Bedrock and the required models.

**3. No Regions Found**
```
Warning: No regions found meeting criteria
```
**Solution:** Try increasing the `max_interruption_rate` parameter or check if the instance types are available.

**4. Cache Issues**
```python
# Clear cache if you're getting stale data
analyzer = SpotPriceAnalyzer()
analyzer.clear_cache()

# Force refresh to bypass cache
results = analyzer.analyze_spot_prices_json(force_refresh=True)
```

### Debug Mode
```bash
# Enable debug logging for troubleshooting
LOG_LEVEL=DEBUG python src/main.py
```

## Performance Tips

1. **Use Caching**: The system caches data for 1 hour by default. Avoid `force_refresh=True` unless you need real-time data.

2. **Warm Cache**: For better performance in production, warm the cache:
```python
analyzer.warm_analysis_cache()
```

3. **Batch Requests**: If analyzing multiple scenarios, do them in sequence to benefit from caching.

4. **Monitor Cache Hit Rate**: Check cache performance:
```python
cache_info = analyzer.get_analysis_cache_info()
print(f"Hit rate: {cache_info['hit_rate']:.2%}")
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs with `LOG_LEVEL=DEBUG`
3. Check AWS credentials and permissions
4. Verify internet connectivity for web scraping

The system is designed to be robust and will provide helpful error messages to guide you through any issues.
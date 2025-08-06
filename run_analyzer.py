#!/usr/bin/env python3
"""
Simple launcher script for AWS Spot Price Analyzer.
This script can be run directly without module path issues.
"""

from src.main import run_analysis_with_params, run_analysis_json_string

# Get results as dictionary
results_dict = run_analysis_with_params(
    instance_types=["p5.48xlarge"],
    max_interruption_rate=0.04,
    top_count=3,
    include_summary=True
)
print(results_dict)

# Get results as formatted JSON string
results_json = run_analysis_json_string(
    instance_types=["p5en.48xlarge", "p5.48xlarge"],
    max_interruption_rate=0.05,
    top_count=3,
    indent=4
)

print(results_json)
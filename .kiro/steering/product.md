# Product Overview

## AWS Spot Advisor Agent

A Python service that analyzes AWS EC2 spot pricing data to identify the most cost-effective regions for high-performance computing instances (p5en.48xlarge and p5.48xlarge).

### Key Features
- Real-time spot price analysis using AWS Bedrock AgentCore for web scraping
- Intelligent filtering based on interruption rates (<5%)
- Automated ranking and recommendation of top 3 regions
- JSON API responses with structured data
- Built-in caching and data freshness validation

### Target Use Case
Helps users optimize costs for GPU-intensive workloads by identifying the best AWS regions for spot instance deployment based on current pricing and reliability metrics.
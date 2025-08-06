#!/usr/bin/env python3
"""Setup script for AWS Spot Price Analyzer."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aws-spot-price-analyzer",
    version="1.0.0",
    author="AWS Spot Price Analyzer Team",
    description="A Python service that analyzes AWS EC2 spot pricing data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "boto3>=1.26.0",
        "python-dotenv>=0.19.0",
        "pydantic>=1.10.0",
        "tenacity>=8.0.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "moto>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "aws-spot-analyzer=src.main:main",
        ],
    },
)
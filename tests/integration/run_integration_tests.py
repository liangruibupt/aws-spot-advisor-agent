#!/usr/bin/env python3
"""
Integration test runner for AWS Spot Price Analyzer.

This script runs all integration tests and provides a summary of results.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_integration_tests():
    """Run all integration tests and provide summary."""
    print("=" * 80)
    print("AWS Spot Price Analyzer - Integration Test Suite")
    print("=" * 80)
    print()
    
    # Test categories to run
    test_categories = [
        {
            "name": "Bedrock Integration Tests",
            "path": "tests/integration/test_bedrock_integration.py",
            "description": "Tests AWS Bedrock AgentCore connectivity and web scraping"
        },
        {
            "name": "End-to-End Workflow Tests", 
            "path": "tests/integration/test_end_to_end_workflow.py",
            "description": "Tests complete data flow from scraping to results"
        },
        {
            "name": "Error Scenario Tests",
            "path": "tests/integration/test_error_scenarios.py", 
            "description": "Tests error handling and recovery mechanisms"
        }
    ]
    
    total_passed = 0
    total_failed = 0
    total_tests = 0
    
    for category in test_categories:
        print(f"Running {category['name']}...")
        print(f"Description: {category['description']}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Run pytest for this category
        cmd = [
            sys.executable, "-m", "pytest", 
            category["path"],
            "-v", "--tb=short", "--no-header"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse results
            output_lines = result.stdout.split('\n')
            summary_line = [line for line in output_lines if 'passed' in line or 'failed' in line]
            
            if summary_line:
                summary = summary_line[-1]
                print(f"Results: {summary}")
                
                # Extract numbers
                if 'passed' in summary:
                    passed = int(summary.split()[0]) if summary.split()[0].isdigit() else 0
                    total_passed += passed
                    total_tests += passed
                
                if 'failed' in summary:
                    # Look for failed count
                    parts = summary.split()
                    for i, part in enumerate(parts):
                        if 'failed' in part and i > 0 and parts[i-1].isdigit():
                            failed = int(parts[i-1])
                            total_failed += failed
                            total_tests += failed
                            break
            
            print(f"Duration: {duration:.2f} seconds")
            
            if result.returncode != 0:
                print("❌ Some tests failed")
                # Show failed test details
                error_lines = [line for line in output_lines if 'FAILED' in line]
                for error_line in error_lines[:5]:  # Show first 5 failures
                    print(f"  {error_line}")
                if len(error_lines) > 5:
                    print(f"  ... and {len(error_lines) - 5} more failures")
            else:
                print("✅ All tests passed")
                
        except subprocess.TimeoutExpired:
            print("❌ Tests timed out after 5 minutes")
            total_failed += 1
            total_tests += 1
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            total_failed += 1
            total_tests += 1
        
        print()
    
    # Overall summary
    print("=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} ❌")
    
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print()
    
    if total_failed == 0:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("⚠️  Some integration tests failed. Check the output above for details.")
        return 1


def run_specific_test_category(category):
    """Run a specific category of tests."""
    categories = {
        "bedrock": "tests/integration/test_bedrock_integration.py",
        "workflow": "tests/integration/test_end_to_end_workflow.py", 
        "errors": "tests/integration/test_error_scenarios.py"
    }
    
    if category not in categories:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return 1
    
    cmd = [
        sys.executable, "-m", "pytest",
        categories[category],
        "-v", "--tb=short"
    ]
    
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific category
        category = sys.argv[1]
        exit_code = run_specific_test_category(category)
    else:
        # Run all tests
        exit_code = run_integration_tests()
    
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Comprehensive test runner for LangGraph Agent
Runs all tests and provides detailed reporting
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"🧪 {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"Exit Code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.3f}s")
        
        if result.stdout:
            print("\n📤 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\n⚠️  STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        if success:
            print("✅ Command completed successfully")
        else:
            print("❌ Command failed")
        
        return success, result.returncode
        
    except Exception as e:
        print(f"❌ Command execution failed: {e}")
        return False, -1

def install_test_dependencies():
    """Install testing dependencies."""
    print("📦 Installing testing dependencies...")
    
    success, _ = run_command(
        "pip install -r requirements-test.txt",
        "Installing Test Dependencies"
    )
    
    return success

def run_unit_tests():
    """Run unit tests."""
    return run_command(
        "python -m pytest tests/test_langgraph_agent.py -v --tb=short",
        "Running LangGraph Agent Unit Tests"
    )

def run_ui_tests():
    """Run UI component tests."""
    return run_command(
        "python -m pytest tests/test_streamlit_ui.py -v --tb=short",
        "Running Streamlit UI Unit Tests"
    )

def run_integration_tests():
    """Run integration tests."""
    return run_command(
        "python -m pytest tests/test_integration.py -v --tb=short -m integration",
        "Running Integration Tests"
    )

def run_all_tests():
    """Run all tests with coverage."""
    return run_command(
        "python -m pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term",
        "Running All Tests with Coverage"
    )

def run_quick_demo():
    """Run a quick demo to verify basic functionality."""
    print("\n🚀 Running Quick Demo Test...")
    
    success, _ = run_command(
        "python demo_langgraph_agent.py demo",
        "Quick Demo Test"
    )
    
    return success

def check_connections():
    """Check service connections."""
    print("\n🔗 Checking Service Connections...")
    
    try:
        # Test MongoDB connection
        success, _ = run_command(
            "python -c \"from src.services.core_service import CoreService; cs = CoreService(); print('✅ MongoDB connection successful')\"",
            "MongoDB Connection Test"
        )
        
        # Test Voyage AI connection
        success2, _ = run_command(
            "python -c \"from src.services.llm_service import LLMService; ls = LLMService(); print('✅ Voyage AI connection successful')\"",
            "Voyage AI Connection Test"
        )
        
        return success and success2
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def generate_test_report(results):
    """Generate a test report summary."""
    print(f"\n{'='*80}")
    print("📊 TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print(f"\n❌ FAILED TESTS:")
        for i, (success, exit_code) in enumerate(results):
            if not success:
                test_names = [
                    "Unit Tests", "UI Tests", "Integration Tests", 
                    "All Tests", "Quick Demo", "Connection Check"
                ]
                print(f"   - {test_names[i]}: Exit code {exit_code}")
    
    print(f"\n{'='*80}")
    
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED! The system is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False

def main():
    """Main test execution function."""
    print("🚀 LangGraph Agent Comprehensive Test Suite")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not Path("src/langgraph_agent.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected: src/langgraph_agent.py")
        sys.exit(1)
    
    # Install test dependencies
    if not install_test_dependencies():
        print("❌ Failed to install test dependencies")
        sys.exit(1)
    
    # Run all tests
    test_results = []
    
    # Unit tests
    success, exit_code = run_unit_tests()
    test_results.append((success, exit_code))
    
    # UI tests
    success, exit_code = run_ui_tests()
    test_results.append((success, exit_code))
    
    # Integration tests
    success, exit_code = run_integration_tests()
    test_results.append((success, exit_code))
    
    # All tests with coverage
    success, exit_code = run_all_tests()
    test_results.append((success, exit_code))
    
    # Quick demo
    success, exit_code = run_quick_demo()
    test_results.append((success, exit_code))
    
    # Connection check
    success, exit_code = check_connections()
    test_results.append((success, exit_code))
    
    # Generate report
    all_passed = generate_test_report(test_results)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()


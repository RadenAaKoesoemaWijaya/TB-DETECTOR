"""
TB DETECTOR - Comprehensive Testing Suite
Tests for all endpoints and functionality
"""

import requests
import json
import os
import sys
import time
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_test(name, status, message=""):
    status_color = Colors.GREEN if status == "PASS" else Colors.RED if status == "FAIL" else Colors.YELLOW
    print(f"  [{status_color}{status}{Colors.END}] {name}")
    if message:
        print(f"       → {message}")

def check_server_running():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        return True
    except:
        return False

def test_root_endpoint():
    """Test root endpoint serves UI"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TEST_TIMEOUT)
        if response.status_code == 200 and "TB DETECTOR" in response.text:
            return True, "UI served successfully"
        return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_api_docs():
    """Test API docs endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=TEST_TIMEOUT)
        if response.status_code == 200:
            return True, "API docs accessible"
        return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_pipeline_status():
    """Test pipeline status endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/pipeline/status", timeout=TEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            required_keys = ['dataset_uploaded', 'preprocessed', 'training_in_progress']
            if all(key in data for key in required_keys):
                return True, f"Status: {data}"
        return False, "Invalid response structure"
    except Exception as e:
        return False, str(e)

def test_logs_endpoint():
    """Test logs endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/pipeline/logs?limit=10", timeout=TEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if 'logs' in data:
                return True, f"Logs count: {len(data['logs'])}"
        return False, "Invalid response"
    except Exception as e:
        return False, str(e)

def test_static_files():
    """Test static files serving"""
    files_to_test = [
        "/static/app_v3.js",
        "/static/preprocessing.js"
    ]
    results = []
    for file in files_to_test:
        try:
            response = requests.get(f"{BASE_URL}{file}", timeout=TEST_TIMEOUT)
            results.append((file, response.status_code == 200))
        except:
            results.append((file, False))
    
    all_pass = all(r[1] for r in results)
    message = ", ".join([f"{f}: {'OK' if s else 'FAIL'}" for f, s in results])
    return all_pass, message

def test_training_results():
    """Test training results endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/training/results", timeout=TEST_TIMEOUT)
        if response.status_code == 200:
            return True, "Training results endpoint accessible"
        return False, f"Status: {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_imports():
    """Test Python imports"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app.models.backbones import BackboneFactory
        from app.models.classifier import TBClassifier
        from app.model_manager import ModelManager
        return True, "All core imports successful"
    except Exception as e:
        return False, str(e)

def test_directory_structure():
    """Test required directories exist"""
    required_dirs = [
        "app/models/weights",
        "data",
        "app/static"
    ]
    results = []
    for dir_path in required_dirs:
        exists = os.path.isdir(dir_path)
        results.append((dir_path, exists))
    
    all_pass = all(r[1] for r in results)
    message = ", ".join([f"{d}: {'OK' if s else 'MISSING'}" for d, s in results])
    return all_pass, message

def test_file_structure():
    """Test required files exist"""
    required_files = [
        "app/main_v3.py",
        "app/model_manager.py",
        "app/static/index_v3.html",
        "app/static/app_v3.js",
        "requirements.txt",
        "README_v3.md"
    ]
    results = []
    for file_path in required_files:
        exists = os.path.isfile(file_path)
        results.append((file_path, exists))
    
    all_pass = all(r[1] for r in results)
    message = ", ".join([f"{f}: {'OK' if s else 'MISSING'}" for f, s in results])
    return all_pass, message

def run_all_tests():
    """Run all tests"""
    print_header("TB DETECTOR - COMPREHENSIVE TEST SUITE")
    
    tests = []
    
    # Pre-flight checks
    print(f"{Colors.YELLOW}PRE-FLIGHT CHECKS{Colors.END}")
    
    # Test 1: Directory structure
    status, msg = test_directory_structure()
    print_test("Directory Structure", "PASS" if status else "FAIL", msg)
    tests.append(("Directory Structure", status))
    
    # Test 2: File structure
    status, msg = test_file_structure()
    print_test("File Structure", "PASS" if status else "FAIL", msg)
    tests.append(("File Structure", status))
    
    # Test 3: Python imports
    status, msg = test_imports()
    print_test("Python Imports", "PASS" if status else "FAIL", msg)
    tests.append(("Python Imports", status))
    
    # Check server
    server_running = check_server_running()
    if not server_running:
        print(f"\n{Colors.YELLOW}⚠️  Server not running. Skipping API tests.{Colors.END}")
        print(f"{Colors.YELLOW}   Start server with: start_v3.bat{Colors.END}\n")
    else:
        print(f"\n{Colors.GREEN}✓ Server detected at {BASE_URL}{Colors.END}\n")
        
        print(f"{Colors.YELLOW}API ENDPOINT TESTS{Colors.END}")
        
        # Test 4: Root endpoint
        status, msg = test_root_endpoint()
        print_test("Root Endpoint (UI)", "PASS" if status else "FAIL", msg)
        tests.append(("Root Endpoint", status))
        
        # Test 5: API Docs
        status, msg = test_api_docs()
        print_test("API Docs", "PASS" if status else "FAIL", msg)
        tests.append(("API Docs", status))
        
        # Test 6: Pipeline status
        status, msg = test_pipeline_status()
        print_test("Pipeline Status", "PASS" if status else "FAIL", msg)
        tests.append(("Pipeline Status", status))
        
        # Test 7: Logs endpoint
        status, msg = test_logs_endpoint()
        print_test("Logs Endpoint", "PASS" if status else "FAIL", msg)
        tests.append(("Logs Endpoint", status))
        
        # Test 8: Static files
        status, msg = test_static_files()
        print_test("Static Files", "PASS" if status else "FAIL", msg)
        tests.append(("Static Files", status))
        
        # Test 9: Training results
        status, msg = test_training_results()
        print_test("Training Results", "PASS" if status else "FAIL", msg)
        tests.append(("Training Results", status))
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, status in tests if status)
    total = len(tests)
    failed = total - passed
    
    print(f"  Total Tests: {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
    
    if failed == 0:
        print(f"\n  {Colors.GREEN}✅ ALL TESTS PASSED!{Colors.END}")
        print(f"  {Colors.GREEN}TB DETECTOR v3 is ready to use.{Colors.END}\n")
        return 0
    else:
        print(f"\n  {Colors.RED}⚠️  SOME TESTS FAILED{Colors.END}")
        print(f"  {Colors.YELLOW}Please check the errors above.{Colors.END}\n")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

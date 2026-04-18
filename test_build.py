"""
Test build untuk TB Detector v3.2
Verifikasi semua imports berfungsi tanpa error
"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports"""
    print("Testing basic imports...")
    try:
        import numpy as np
        import pandas as pd
        import torch
        import fastapi
        import librosa
        import matplotlib
        print("✓ Basic dependencies OK")
        return True
    except ImportError as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_app_imports():
    """Test app module imports"""
    print("\nTesting app module imports...")
    
    tests = [
        ("app.models.backbones", "BackboneFactory", "Backbone factory"),
        ("app.models.classifier", "TBClassifier", "Classifier"),
        ("app.models.preprocessing", "AudioPreprocessor", "Preprocessor"),
        ("app.utils.feature_fusion", "SimpleConcatFusion", "Feature fusion"),
        ("app.utils.metadata_encoder", "SimpleMetadataEncoder", "Metadata encoder"),
        ("app.model_manager", "get_model_manager", "Model manager"),
        ("app.persistence", "get_persistence", "Persistence"),
    ]
    
    all_passed = True
    for module, name, desc in tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            print(f"  ✓ {desc}")
        except Exception as e:
            print(f"  ✗ {desc}: {e}")
            all_passed = False
    
    return all_passed

def test_phase1_imports():
    """Test Phase 1 training imports"""
    print("\nTesting Phase 1 imports...")
    
    tests = [
        ("app.training", "BatchTrainer", "Batch trainer"),
        ("app.training", "get_feature_cache", "Feature cache"),
    ]
    
    all_passed = True
    for module, name, desc in tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            print(f"  ✓ {desc}")
        except Exception as e:
            print(f"  ✗ {desc}: {e}")
            all_passed = False
    
    return all_passed

def test_phase2_imports():
    """Test Phase 2 imports"""
    print("\nTesting Phase 2 imports...")
    
    tests = [
        ("app.async_utils", "get_async_io", "Async I/O"),
        ("app.task_queue", "get_task_queue", "Task queue"),
        ("app.task_queue", "TaskType", "Task types"),
        ("app.onnx_inference", "ONNX_AVAILABLE", "ONNX available flag"),
        ("app.onnx_inference", "get_inference_manager", "Inference manager"),
        ("app.model_versioning", "get_model_registry", "Model registry"),
        ("app.model_versioning", "ModelStage", "Model stages"),
        ("app.ab_testing", "get_ab_testing", "A/B testing"),
        ("app.ab_testing", "ExperimentStatus", "Experiment status"),
    ]
    
    all_passed = True
    for module, name, desc in tests:
        try:
            mod = __import__(module, fromlist=[name])
            getattr(mod, name)
            print(f"  ✓ {desc}")
        except Exception as e:
            print(f"  ✗ {desc}: {e}")
            all_passed = False
    
    return all_passed

def test_main_v3_import():
    """Test main_v3 can be imported"""
    print("\nTesting main_v3 import...")
    try:
        # We can't fully import main_v3 due to FastAPI app initialization,
        # but we can check the file is syntactically correct by compiling it
        with open("app/main_v3.py", "r") as f:
            code = f.read()
        compile(code, "app/main_v3.py", "exec")
        print("  ✓ main_v3.py syntax OK")
        return True
    except SyntaxError as e:
        print(f"  ✗ main_v3.py syntax error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ main_v3.py error: {e}")
        return False

def main():
    """Run all build tests"""
    print("=" * 60)
    print("TB Detector v3.2 - Build Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Basic Dependencies", test_basic_imports()))
    results.append(("App Modules", test_app_imports()))
    results.append(("Phase 1 Modules", test_phase1_imports()))
    results.append(("Phase 2 Modules", test_phase2_imports()))
    results.append(("Main v3 Syntax", test_main_v3_import()))
    
    print("\n" + "=" * 60)
    print("Build Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Build successful! Application ready to run.")
        print("\nTo start the server:")
        print("  .\\start_v3.bat")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

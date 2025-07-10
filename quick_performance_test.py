#!/usr/bin/env python3
"""
Quick Performance Test for FTT
==============================
A simple script to test basic performance of key FTT components.
"""

import time
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def time_function(func, *args, **kwargs):
    """Time a function execution"""
    start = time.time()
    try:
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start, None
    except Exception as e:
        end = time.time()
        return None, end - start, str(e)

def test_imports():
    """Test import times"""
    print("Testing import performance...")
    
    tests = [
        ("numpy", lambda: __import__('numpy')),
        ("pandas", lambda: __import__('pandas')),
        ("bottle", lambda: __import__('bottle')),
        ("ModelRun", lambda: __import__('SourceCode.model_class', fromlist=['ModelRun'])),
    ]
    
    for name, import_func in tests:
        result, duration, error = time_function(import_func)
        if error:
            print(f"  ❌ {name}: Failed ({error:.50}...)")
        else:
            print(f"  ✅ {name}: {duration:.3f}s")

def test_model_creation():
    """Test model creation performance"""
    print("\nTesting model creation...")
    
    try:
        from SourceCode.model_class import ModelRun
        
        result, duration, error = time_function(ModelRun)
        if error:
            print(f"  ❌ ModelRun creation: Failed ({error})")
        else:
            print(f"  ✅ ModelRun creation: {duration:.3f}s")
            return result
    except Exception as e:
        print(f"  ❌ Cannot import ModelRun: {e}")
        return None

def test_backend_import():
    """Test backend import performance"""
    print("\nTesting backend import...")
    
    result, duration, error = time_function(lambda: __import__('Backend_FTT'))
    if error:
        print(f"  ❌ Backend_FTT import: Failed ({error:.100}...)")
    else:
        print(f"  ✅ Backend_FTT import: {duration:.3f}s")

def main():
    """Run quick performance tests"""
    print("FTT Quick Performance Test")
    print("=" * 40)
    
    # Test imports
    test_imports()
    
    # Test model creation
    model = test_model_creation()
    
    # Test backend import
    test_backend_import()
    
    print("\n" + "=" * 40)
    print("Quick test completed!")
    print("\nFor detailed profiling, run:")
    print("  python profile_ftt.py")
    print("  python memory_profiler.py")

if __name__ == "__main__":
    main()

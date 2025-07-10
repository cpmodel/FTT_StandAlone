#!/usr/bin/env python3
"""
Line-by-line profiler for FTT Stand Alone
=========================================
This script provides detailed line-by-line profiling for specific functions.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def profile_specific_functions():
    """
    Profile specific functions with line_profiler
    
    Usage:
    1. Add @profile decorator to functions you want to profile
    2. Run: kernprof -l -v line_profiler_ftt.py
    """
    
    print("Setting up line-by-line profiling...")
    
    try:
        # Import your modules
        from SourceCode.model_class import ModelRun
        
        # Create instance
        model = ModelRun()
        
        # Example: Profile a specific method
        # You'll need to add @profile decorator to the method in the source
        print("Model instance created successfully")
        print("To use line profiler:")
        print("1. Add @profile decorator to functions in your source code")
        print("2. Run: kernprof -l -v line_profiler_ftt.py")
        print("3. View results with: python -m line_profiler line_profiler_ftt.py.lprof")
        
    except Exception as e:
        print(f"Error: {e}")

# Example of how to add profiling decorators to your functions
def example_function_to_profile():
    """
    Example function showing how to use @profile decorator
    
    To profile this function:
    1. Uncomment the @profile decorator below
    2. Run: kernprof -l -v line_profiler_ftt.py
    """
    
    # @profile  # Uncomment this line to profile this function
    def heavy_computation():
        import numpy as np
        
        # Simulate heavy computation
        data = np.random.rand(1000, 1000)
        result = np.dot(data, data.T)
        eigenvals = np.linalg.eigvals(result)
        return eigenvals
    
    return heavy_computation()

def create_profiling_template():
    """Create a template for adding profiling to your existing functions"""
    
    template = '''
# Profiling Template for FTT Functions
# ===================================

# 1. Add this import at the top of your Python file:
# import line_profiler

# 2. Add @profile decorator before functions you want to profile:

@profile
def your_function_name(self, *args, **kwargs):
    """Your function docstring"""
    # Your existing code here
    pass

# 3. Run profiling with:
# kernprof -l -v your_script.py

# 4. View detailed results:
# python -m line_profiler your_script.py.lprof

# Example for FTT model_class.py:
# Add @profile before methods like:
# - solve_year()
# - load_data()
# - Any computationally expensive methods
'''
    
    # Save template
    template_path = Path("./Output/Profiles/profiling_template.txt")
    template_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(template_path, 'w') as f:
        f.write(template)
    
    print(f"Profiling template saved to: {template_path}")
    print("Use this template to add profiling to your FTT functions")

if __name__ == "__main__":
    profile_specific_functions()
    create_profiling_template()
    
    # Run example
    print("\nRunning example computation...")
    result = example_function_to_profile()
    print(f"Example completed. Result shape: {len(result) if hasattr(result, '__len__') else 'scalar'}")

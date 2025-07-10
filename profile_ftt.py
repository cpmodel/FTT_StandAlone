#!/usr/bin/env python3
"""
Profile FTT Stand Alone Model
============================
This script profiles the FTT model to identify performance bottlenecks.
"""

import cProfile
import pstats
import io
from pathlib import Path
import sys
import time

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def profile_ftt_backend():
    """Profile the Backend_FTT.py main execution"""
    print("Profiling Backend_FTT.py...")
    
    # Create a profiler
    profiler = cProfile.Profile()
    
    # Start profiling
    profiler.enable()
    
    try:
        # Import and run your backend (modify as needed)
        import Backend_FTT
        # If you have specific functions to test, call them here
        # For example: Backend_FTT.some_function()
        
    except Exception as e:
        print(f"Error during profiling: {e}")
    finally:
        profiler.disable()
    
    return profiler

def profile_model_run():
    """Profile a complete model run"""
    print("Profiling ModelRun class...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        from SourceCode.model_class import ModelRun
        
        # Create model instance
        model = ModelRun()
        
        # Profile a short simulation (modify timeline for testing)
        # This is where the heavy computation happens
        test_scenario = "S0"  # Use a baseline scenario
        
        # Profile just the initialization
        print("Profiling model initialization...")
        
    except Exception as e:
        print(f"Error during model profiling: {e}")
    finally:
        profiler.disable()
    
    return profiler

def profile_specific_function(func, *args, **kwargs):
    """Profile a specific function"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        profiler.disable()
        return profiler

def analyze_profile(profiler, output_file=None, top_n=20):
    """Analyze and display profiling results"""
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    # Sort by cumulative time
    ps.sort_stats('cumulative')
    ps.print_stats(top_n)
    
    # Also sort by time spent in function itself
    print("\n" + "="*50)
    print("TOP FUNCTIONS BY SELF TIME:")
    print("="*50)
    ps.sort_stats('time')
    ps.print_stats(top_n)
    
    # Save to file if requested
    if output_file:
        ps.dump_stats(output_file)
        print(f"\nProfile saved to: {output_file}")
    
    # Print summary
    profile_output = s.getvalue()
    print(profile_output)
    
    return profile_output

def main():
    """Main profiling function"""
    print("FTT Stand Alone Model Profiler")
    print("="*40)
    
    # Create output directory for profiles
    profile_dir = Path("./Output/Profiles")
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    # Profile different components
    components_to_profile = [
        ("backend", profile_ftt_backend),
        ("model_run", profile_model_run),
    ]
    
    for name, profile_func in components_to_profile:
        print(f"\n{'='*20} PROFILING {name.upper()} {'='*20}")
        
        start_time = time.time()
        profiler = profile_func()
        end_time = time.time()
        
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
        # Analyze results
        output_file = profile_dir / f"{name}_profile.prof"
        analyze_profile(profiler, output_file)
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()

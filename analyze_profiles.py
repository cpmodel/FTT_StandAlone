#!/usr/bin/env python3
"""
Analyze FTT Profiling Results
============================
This script analyzes the profiling results and provides actionable insights.
"""

import pstats
import io
from pathlib import Path

def analyze_profile_file(profile_path, top_n=15):
    """Analyze a specific profile file"""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {profile_path.name}")
    print(f"{'='*60}")
    
    # Load the profile
    stats = pstats.Stats(str(profile_path))
    
    # Create string buffer for output
    s = io.StringIO()
    stats.stream = s
    
    # Sort by cumulative time (most important)
    print("\n🕒 TOP FUNCTIONS BY CUMULATIVE TIME:")
    print("-" * 40)
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)
    output = s.getvalue()
    print(output[-2000:])  # Print last 2000 chars to avoid too much output
    
    # Reset buffer
    s = io.StringIO()
    stats.stream = s
    
    # Sort by self time (time spent in function itself)
    print("\n⚡ TOP FUNCTIONS BY SELF TIME:")
    print("-" * 40)
    stats.sort_stats('time')
    stats.print_stats(top_n)
    output = s.getvalue()
    print(output[-2000:])
    
    # Get caller information for top functions
    s = io.StringIO()
    stats.stream = s
    
    print("\n📞 CALLER ANALYSIS (Top 5):")
    print("-" * 40)
    stats.sort_stats('cumulative')
    stats.print_callers(5)
    output = s.getvalue()
    print(output[-1500:])
    
    return stats

def provide_optimization_recommendations(backend_stats, model_stats):
    """Provide specific optimization recommendations"""
    print(f"\n{'='*60}")
    print("🚀 OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print("\n1. 📊 IMPORT OPTIMIZATION:")
    print("   - Backend imports take ~0.97s (mostly pandas/numba)")
    print("   - Consider lazy imports for heavy modules")
    print("   - Move imports closer to where they're used")
    
    print("\n2. 🏗️ MODEL INITIALIZATION:")
    print("   - ModelRun creation takes ~20+ seconds")
    print("   - This suggests heavy data loading")
    print("   - Consider caching preprocessed data")
    print("   - Profile data loading functions specifically")
    
    print("\n3. 📈 DATA PROCESSING:")
    print("   - Focus on pandas operations in data loading")
    print("   - Use vectorized operations instead of loops")
    print("   - Consider using numpy directly for numerical operations")
    
    print("\n4. 🔄 MEMORY USAGE:")
    print("   - Run memory_profiler.py to check for memory leaks")
    print("   - Large datasets might benefit from chunking")
    
    print("\n5. ⚡ NUMBA OPTIMIZATION:")
    print("   - Consider @numba.jit decorators for computation-heavy functions")
    print("   - Focus on yearly solving loop (solve_year function)")
    
    print("\n6. 🗂️ FILE I/O:")
    print("   - Excel/CSV reading appears to be a bottleneck")
    print("   - Consider converting to more efficient formats (parquet, HDF5)")
    print("   - Cache processed data to avoid re-reading")

def identify_bottlenecks():
    """Identify specific bottlenecks from the output"""
    print(f"\n{'='*60}")
    print("🎯 IDENTIFIED BOTTLENECKS")
    print(f"{'='*60}")
    
    bottlenecks = [
        {
            "area": "Pandas Import",
            "time": "~0.45s",
            "impact": "Medium",
            "fix": "Move to lazy import or use alternatives"
        },
        {
            "area": "Model Class Import", 
            "time": "~0.33s",
            "impact": "Medium",
            "fix": "Optimize module structure, reduce dependencies"
        },
        {
            "area": "Numba Import",
            "time": "~0.26s", 
            "impact": "Low",
            "fix": "Consider numba.optional for conditional import"
        },
        {
            "area": "ModelRun Creation",
            "time": "20+ seconds",
            "impact": "HIGH", 
            "fix": "Profile data loading, implement caching"
        }
    ]
    
    for i, bottleneck in enumerate(bottlenecks, 1):
        impact_emoji = {"HIGH": "🔴", "Medium": "🟡", "Low": "🟢"}
        print(f"\n{i}. {impact_emoji[bottleneck['impact']]} {bottleneck['area']}")
        print(f"   Time: {bottleneck['time']}")
        print(f"   Impact: {bottleneck['impact']}")
        print(f"   Fix: {bottleneck['fix']}")

def generate_next_steps():
    """Generate actionable next steps"""
    print(f"\n{'='*60}")
    print("📋 NEXT STEPS")
    print(f"{'='*60}")
    
    steps = [
        "1. Profile ModelRun.__init__() with line_profiler",
        "2. Add @profile to data loading functions", 
        "3. Run memory_profiler.py to check memory usage",
        "4. Profile solve_year() function specifically",
        "5. Consider converting Excel files to parquet format",
        "6. Implement data caching for repeated runs",
        "7. Add @numba.jit to numerical computation functions",
        "8. Profile pandas operations in input_functions.py"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n💡 PRIORITY:")
    print("   Focus on ModelRun data loading (20s is significant)")
    print("   This will give the biggest performance improvement")

def main():
    """Main analysis function"""
    profile_dir = Path("./Output/Profiles")
    
    print("FTT PROFILING RESULTS ANALYSIS")
    print("=" * 60)
    
    # Analyze backend profile
    backend_file = profile_dir / "backend_profile.prof"
    if backend_file.exists():
        backend_stats = analyze_profile_file(backend_file)
    else:
        print("Backend profile not found!")
        backend_stats = None
    
    # Analyze model profile  
    model_file = profile_dir / "model_run_profile.prof"
    if model_file.exists():
        model_stats = analyze_profile_file(model_file)
    else:
        print("Model profile not found!")
        model_stats = None
    
    # Provide recommendations
    provide_optimization_recommendations(backend_stats, model_stats)
    
    # Identify bottlenecks
    identify_bottlenecks()
    
    # Generate next steps
    generate_next_steps()
    
    print(f"\n{'='*60}")
    print("📁 PROFILE FILES LOCATION:")
    print(f"   {profile_dir.absolute()}")
    print("   Use snakeviz for interactive analysis:")
    print("   pip install snakeviz")
    print("   snakeviz Output/Profiles/backend_profile.prof")

if __name__ == "__main__":
    main()

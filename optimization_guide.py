#!/usr/bin/env python3
"""
FTT Performance Optimization Guide
==================================
Non-data optimization strategies for FTT Stand Alone model.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def optimize_numpy_operations():
    """
    Optimization strategies for numpy operations in FTT
    """
    print("🚀 NUMPY OPTIMIZATION STRATEGIES")
    print("=" * 50)
    
    strategies = [
        {
            "name": "Use appropriate dtypes",
            "before": "data = np.array([1, 2, 3])  # defaults to int64/float64",
            "after": "data = np.array([1, 2, 3], dtype=np.float32)  # Use float32 if precision allows",
            "benefit": "50% memory reduction, faster operations"
        },
        {
            "name": "Vectorize operations",
            "before": """
# Slow: Python loop
result = np.zeros(len(data))
for i in range(len(data)):
    result[i] = np.exp(data[i]) / (1 + np.exp(data[i]))
            """,
            "after": """
# Fast: Vectorized
result = np.exp(data) / (1 + np.exp(data))
# or even better: use scipy.special.expit
from scipy.special import expit
result = expit(data)
            """,
            "benefit": "10-100x speedup depending on data size"
        },
        {
            "name": "Use broadcasting",
            "before": """
# Slow: explicit loops
result = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):
        result[i, j] = array1[i] * array2[j]
            """,
            "after": """
# Fast: broadcasting
result = array1[:, np.newaxis] * array2[np.newaxis, :]
            """,
            "benefit": "Massive speedup, cleaner code"
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   Benefit: {strategy['benefit']}")
        print(f"   Before: {strategy['before']}")
        print(f"   After: {strategy['after']}")

def optimize_pandas_operations():
    """
    Optimization strategies for pandas operations in FTT
    """
    print("\n\n🐼 PANDAS OPTIMIZATION STRATEGIES")
    print("=" * 50)
    
    strategies = [
        {
            "name": "Use .loc/.iloc instead of chained indexing",
            "before": "df[df['column'] > 5]['other_column'].values",
            "after": "df.loc[df['column'] > 5, 'other_column'].values",
            "benefit": "Avoids SettingWithCopyWarning, faster"
        },
        {
            "name": "Use categorical data for repeated strings",
            "before": "df['category'] = df['category']  # string type",
            "after": "df['category'] = df['category'].astype('category')",
            "benefit": "Significant memory reduction, faster operations"
        },
        {
            "name": "Use vectorized string operations",
            "before": "df['result'] = df['text'].apply(lambda x: x.upper())",
            "after": "df['result'] = df['text'].str.upper()",
            "benefit": "Faster string operations"
        },
        {
            "name": "Use query() for complex filtering",
            "before": "result = df[(df['A'] > 5) & (df['B'] < 10) & (df['C'] == 'value')]",
            "after": "result = df.query('A > 5 and B < 10 and C == \"value\"')",
            "benefit": "More readable, sometimes faster"
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   Benefit: {strategy['benefit']}")
        print(f"   Before: {strategy['before']}")
        print(f"   After: {strategy['after']}")

def optimize_computational_patterns():
    """
    General computational optimization patterns for FTT
    """
    print("\n\n⚡ COMPUTATIONAL OPTIMIZATION PATTERNS")
    print("=" * 50)
    
    patterns = [
        {
            "name": "Cache expensive computations",
            "description": "Store results of expensive calculations",
            "example": """
# Add caching to expensive functions
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_calculation(param1, param2):
    # expensive computation here
    return result
            """
        },
        {
            "name": "Use numba for hot loops",
            "description": "JIT compile numerical functions",
            "example": """
from numba import jit

@jit(nopython=True)
def fast_numerical_function(array):
    result = 0.0
    for i in range(len(array)):
        result += array[i] ** 2
    return result
            """
        },
        {
            "name": "Parallelize independent operations",
            "description": "Use multiprocessing for CPU-bound tasks",
            "example": """
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def process_chunk(chunk):
    # Process each chunk independently
    return expensive_operation(chunk)

# Split work into chunks
chunks = np.array_split(large_array, n_processes)
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_chunk, chunks))
            """
        }
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\n{i}. {pattern['name']}")
        print(f"   Description: {pattern['description']}")
        print(f"   Example: {pattern['example']}")

def analyze_ftt_specific_optimizations():
    """
    FTT-specific optimization opportunities
    """
    print("\n\n🎯 FTT-SPECIFIC OPTIMIZATION OPPORTUNITIES")
    print("=" * 50)
    
    optimizations = [
        {
            "area": "Model Solve Loop",
            "file": "SourceCode/model_class.py",
            "function": "solve_year()",
            "opportunities": [
                "Pre-allocate arrays for yearly results",
                "Minimize dictionary lookups in tight loops",
                "Use numpy views instead of copying arrays",
                "Cache frequently accessed model parameters"
            ]
        },
        {
            "area": "Transport Model",
            "file": "SourceCode/Transport/",
            "function": "Various ftt_tr_* functions",
            "opportunities": [
                "Vectorize survival function calculations",
                "Use sparse matrices for large, mostly-zero arrays",
                "Pre-compute common mathematical operations",
                "Optimize LCOT (Levelized Cost of Transport) calculations"
            ]
        },
        {
            "area": "Power Model", 
            "file": "SourceCode/Power/",
            "function": "Various ftt_p_* functions",
            "opportunities": [
                "Optimize RLDC (Residual Load Duration Curve) calculations",
                "Use efficient matrix operations for power shares",
                "Cache capacity factor calculations",
                "Vectorize power plant survival functions"
            ]
        },
        {
            "area": "Backend API",
            "file": "Backend_FTT.py",
            "function": "retrieve_chart_data()",
            "opportunities": [
                "Cache processed results",
                "Optimize pandas pivot operations",
                "Use more efficient JSON serialization",
                "Pre-compute common aggregations"
            ]
        }
    ]
    
    for opt in optimizations:
        print(f"\n📍 {opt['area']}")
        print(f"   File: {opt['file']}")
        print(f"   Function: {opt['function']}")
        print("   Optimization opportunities:")
        for opp in opt['opportunities']:
            print(f"   • {opp}")

def optimization_checklist():
    """Print optimization checklist"""
    print("FTT PERFORMANCE OPTIMIZATION CHECKLIST")
    print("=" * 50)
    
    optimizations = [
        {
            "category": "🔴 CRITICAL - Data Loading (38.4s)",
            "items": [
                "✅ Convert CSV files to Parquet format (5-10x faster)",
                "✅ Implement data caching (save processed data)",
                "✅ Use chunked reading for large files",
                "✅ Consider HDF5 for numerical data",
                "✅ Add progress bars to data loading"
            ]
        },
        {
            "category": "🟡 HIGH - Pandas Operations", 
            "items": [
                "✅ Replace .loc/.iloc with direct numpy operations where possible",
                "✅ Use .query() instead of boolean indexing",
                "✅ Vectorize operations instead of iterating",
                "✅ Consider polars for faster dataframes",
                "✅ Use categorical data for repeated strings"
            ]
        },
        {
            "category": "🟢 MEDIUM - Code Structure",
            "items": [
                "✅ Add @numba.jit to numerical functions",
                "✅ Implement lazy imports",
                "✅ Cache frequently accessed data",
                "✅ Use multiprocessing for independent operations",
                "✅ Profile solve_year() function specifically"
            ]
        }
    ]
    
    for opt in optimizations:
        print(f"\n{opt['category']}")
        print("-" * 40)
        for item in opt['items']:
            print(f"  {item}")

def create_parquet_converter():
    """Create a script to convert CSV to Parquet"""
    script = '''#!/usr/bin/env python3
"""
Convert FTT CSV files to Parquet for faster loading
==================================================
"""
import pandas as pd
from pathlib import Path
import time

def convert_csv_to_parquet(input_dir, output_dir):
    """Convert all CSV files in directory to Parquet"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(input_path.glob("**/*.csv"))
    print(f"Found {len(csv_files)} CSV files to convert")
    
    for csv_file in csv_files:
        try:
            # Read CSV
            start = time.time()
            df = pd.read_csv(csv_file)
            
            # Create corresponding parquet path
            rel_path = csv_file.relative_to(input_path)
            parquet_file = output_path / rel_path.with_suffix('.parquet')
            parquet_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet
            df.to_parquet(parquet_file, compression='snappy')
            
            end = time.time()
            print(f"✅ {csv_file.name}: {end-start:.2f}s")
            
        except Exception as e:
            print(f"❌ {csv_file.name}: {e}")

if __name__ == "__main__":
    # Convert your inputs
    convert_csv_to_parquet("./Inputs", "./Inputs_Parquet")
    convert_csv_to_parquet("./Utilities", "./Utilities_Parquet")
'''
    
    with open("convert_to_parquet.py", "w") as f:
        f.write(script)
    
    print("\n📁 Created: convert_to_parquet.py")
    print("   Run this to convert your CSV files to faster Parquet format")

def create_caching_example():
    """Create an example caching implementation"""
    script = '''#!/usr/bin/env python3
"""
Example caching implementation for FTT data loading
==================================================
"""
import pickle
import hashlib
from pathlib import Path
import time

class DataCache:
    """Simple data caching system"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, file_path, params=None):
        """Generate cache key from file path and parameters"""
        content = str(file_path)
        if params:
            content += str(sorted(params.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_cached(self, cache_key):
        """Check if data is cached"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def load_from_cache(self, cache_key):
        """Load data from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def save_to_cache(self, cache_key, data):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

def cached_load_data(file_path, cache=None):
    """Example of cached data loading"""
    if cache is None:
        cache = DataCache()
    
    cache_key = cache.get_cache_key(file_path)
    
    if cache.is_cached(cache_key):
        print(f"📁 Loading {file_path.name} from cache...")
        return cache.load_from_cache(cache_key)
    
    print(f"📂 Loading {file_path.name} from disk...")
    # Your normal loading logic here
    import pandas as pd
    data = pd.read_csv(file_path)  # or read_parquet
    
    # Cache the result
    cache.save_to_cache(cache_key, data)
    return data

# Example usage:
# cache = DataCache()
# data = cached_load_data(Path("some_file.csv"), cache)
'''
    
    with open("data_caching_example.py", "w") as f:
        f.write(script)
    
    print("📁 Created: data_caching_example.py")
    print("   Example implementation of data caching")

def immediate_wins():
    """List immediate performance wins"""
    print("\n🚀 IMMEDIATE PERFORMANCE WINS:")
    print("-" * 40)
    
    wins = [
        "1. Replace pd.read_csv() with pd.read_parquet() - 5-10x faster",
        "2. Add caching to avoid re-reading unchanged files",  
        "3. Use pd.read_csv(low_memory=False) to avoid dtype inference",
        "4. Add progress bars: from tqdm import tqdm",
        "5. Profile input_functions.py line-by-line",
        "6. Consider dask for out-of-core processing of large datasets"
    ]
    
    for win in wins:
        print(f"  {win}")
    
    print(f"\n💡 EXPECTED IMPROVEMENT:")
    print("  Current: 38.4s data loading")
    print("  With parquet: ~5-8s")
    print("  With caching: ~1-2s (after first load)")
    print("  Total speedup: 10-40x faster! 🚀")

def main():
    """Main function"""
    optimization_checklist()
    print("\n")
    create_parquet_converter()
    print("\n")
    create_caching_example()
    immediate_wins()
    
    print(f"\n{'='*50}")
    print("📋 NEXT ACTIONS:")
    print("1. Run: python convert_to_parquet.py")
    print("2. Modify input_functions.py to use parquet files")
    print("3. Implement caching for preprocessed data") 
    print("4. Add @profile decorator to load_data() function")
    print("5. Run line profiler on input_functions.py")

if __name__ == "__main__":
    main()

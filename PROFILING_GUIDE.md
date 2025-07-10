# FTT Stand Alone Profiling Guide
================================

This guide provides multiple approaches to profile your FTT (Future Technology Transformations) model for performance optimization.

## Quick Start

### 1. Basic Performance Profiling
```bash
# Run the basic profiler
python profile_ftt.py
```

### 2. Memory Usage Analysis
```bash
# Monitor memory usage
python memory_profiler.py
```

### 3. Line-by-Line Profiling
```bash
# Setup line profiling (requires adding @profile decorators)
python line_profiler_ftt.py
```

## Profiling Methods Explained

### Method 1: cProfile (Built-in, Start Here)
**Best for**: Overall performance overview, finding bottlenecks

**Files**: `profile_ftt.py`

**What it shows**:
- Function call counts
- Time spent in each function
- Cumulative time including sub-functions

**Usage**:
1. Run: `python profile_ftt.py`
2. Check `./Output/Profiles/` for detailed reports
3. Look for functions with high cumulative time

### Method 2: Memory Profiling
**Best for**: Memory leaks, memory-intensive operations

**Files**: `memory_profiler.py`

**What it shows**:
- Memory usage over time
- Peak memory consumption
- Memory allocation patterns

**Usage**:
1. Run: `python memory_profiler.py`
2. Check generated plots in `./Output/Profiles/`

### Method 3: Line-by-Line Profiling
**Best for**: Detailed analysis of specific functions

**Requirements**: Add `@profile` decorator to functions

**What it shows**:
- Time spent on each line of code
- Number of times each line executes

**Setup**:
1. Add `@profile` decorator to functions in your source code
2. Run: `kernprof -l -v your_script.py`
3. View: `python -m line_profiler your_script.py.lprof`

## Recommended Profiling Strategy for FTT

### Phase 1: High-Level Analysis
1. **Run cProfile first**: `python profile_ftt.py`
2. **Identify top 10 slowest functions**
3. **Check memory usage**: `python memory_profiler.py`

### Phase 2: Deep Dive
1. **Add @profile decorators** to slowest functions
2. **Use line profiler** on specific bottlenecks
3. **Focus on numpy/pandas operations** (common bottlenecks)

### Phase 3: Optimization
1. **Vectorize operations** where possible
2. **Optimize data loading** and preprocessing
3. **Consider caching** expensive computations

## Key Areas to Profile in FTT (Computational Focus)

Based on your responsibilities, focus on:

### 1. Yearly Solving Loop
- **File**: `SourceCode/model_class.py`
- **Function**: `solve_year()`
- **Why**: Called repeatedly, core computation logic

### 2. Transport Model Calculations
- **Files**: `SourceCode/Transport/ftt_tr_*.py`
- **Why**: Complex numerical calculations, survival functions

### 3. Power Model Calculations
- **Files**: `SourceCode/Power/ftt_p_*.py`
- **Why**: Matrix operations, RLDC calculations

### 4. Mathematical Support Functions
- **Files**: `SourceCode/support/`
- **Why**: Numerical utilities and helper functions

**Note**: Data loading and backend API optimization are handled by other team members.

## Common Performance Issues in Scientific Computing

### 1. Numpy Array Operations
```python
# Slow (Python loops)
for i in range(len(array)):
    result[i] = expensive_function(array[i])

# Fast (vectorized)
result = np.vectorize(expensive_function)(array)
# or even better: use built-in numpy functions
result = np.exp(array)  # if expensive_function is exponential
```

### 2. Mathematical Computations
```python
# Slow (element-wise in loops)
survival = np.zeros(len(ages))
for i, age in enumerate(ages):
    survival[i] = np.exp(-age / lifetime)

# Fast (vectorized)
survival = np.exp(-ages / lifetime)
```

### 3. Memory Allocations
```python
# Slow (growing arrays)
results = []
for year in years:
    result = calculate_year(year)
    results.append(result)

# Fast (pre-allocated)
results = np.zeros((len(years), n_regions, n_techs))
for i, year in enumerate(years):
    results[i] = calculate_year(year)
```

## Interpreting Profile Results

### cProfile Output
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
10      0.003    0.000    0.500    0.050   your_function()
```

- **ncalls**: Number of calls
- **tottime**: Total time excluding sub-functions  
- **cumtime**: Total time including sub-functions
- **Focus on**: High cumtime values

### Memory Profiler
- **Look for**: Sharp increases in memory usage
- **Identify**: Functions that don't release memory
- **Watch for**: Memory leaks in loops

## Next Steps After Profiling

1. **Identify computational bottlenecks** (functions taking >10% of total time)
2. **Optimize numerical algorithms** (use appropriate numpy dtypes)
3. **Vectorize operations** (replace loops with numpy operations)
4. **Add @numba.jit** to numerical hot spots
5. **Cache expensive computations** (avoid recalculating same values)
6. **Pre-allocate arrays** (avoid memory allocations in loops)

## Files Created

- `profile_ftt.py` - Main profiling script
- `memory_profiler.py` - Memory usage monitoring
- `line_profiler_ftt.py` - Line-by-line profiling setup
- `./Output/Profiles/` - Directory for profiling results

## Tools Installation

If you need to install additional profiling tools:
```bash
pip install line_profiler memory_profiler py-spy snakeviz
```

## Visualization Tools

- **snakeviz**: Interactive cProfile viewer
  ```bash
  snakeviz profile_results.prof
  ```

- **py-spy**: Live profiling of running processes
  ```bash
  py-spy top --pid <process_id>
  ```

Remember: Profile first, optimize second. Don't guess where the bottlenecks are!

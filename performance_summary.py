#!/usr/bin/env python3
"""
FTT Performance Summary & Recommendations
=========================================
Based on profiling your FTT model.
"""

def print_summary():
    """Print the performance analysis summary"""
    print("FTT PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 50)
    
    print("\nCRITICAL FINDINGS:")
    print("- Data loading takes 38.4 seconds (!)") 
    print("- Reading 3,688 CSV files with pandas")
    print("- Memory usage: 59 MB -> 267 MB")
    print("- Lots of pandas indexing operations")
    
    print("\nIMMEDIATE OPTIMIZATIONS:")
    print("1. Convert CSV to Parquet format (5-10x faster)")
    print("2. Implement data caching")
    print("3. Add progress bars to see loading progress")
    print("4. Profile input_functions.py line-by-line")
    
    print("\nEXPECTED IMPROVEMENTS:")
    print("- Current: 38.4s data loading")
    print("- With Parquet: ~5-8s")
    print("- With caching: ~1-2s (after first load)")
    print("- Total speedup potential: 10-40x!")

def create_parquet_converter():
    """Create CSV to Parquet converter"""
    converter_code = '''#!/usr/bin/env python3
"""Convert CSV files to Parquet for faster loading"""
import pandas as pd
from pathlib import Path
import time

def convert_directory(input_dir, output_dir):
    """Convert all CSV files to Parquet"""
    input_path = Path(input_dir)
    output_path = Path(output_dir) 
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(input_path.glob("**/*.csv"))
    print(f"Converting {len(csv_files)} CSV files...")
    
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            
            # Create parquet path
            rel_path = csv_file.relative_to(input_path)
            parquet_file = output_path / rel_path.with_suffix('.parquet')
            parquet_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet
            df.to_parquet(parquet_file, compression='snappy')
            
            if i % 100 == 0:
                print(f"Converted {i}/{len(csv_files)} files...")
                
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")
    
    print("Conversion complete!")

if __name__ == "__main__":
    convert_directory("./Inputs", "./Inputs_Parquet")
'''
    
    with open("csv_to_parquet.py", "w", encoding='utf-8') as f:
        f.write(converter_code)
    
    print("\nCreated: csv_to_parquet.py")
    print("Run: python csv_to_parquet.py")

def next_steps():
    """Print next steps"""
    print("\nNEXT STEPS:")
    print("=" * 20)
    print("1. Run: python csv_to_parquet.py")
    print("2. Modify input_functions.py to use .parquet files")
    print("3. Add @profile decorator to load_data() function")  
    print("4. Run: kernprof -l -v SourceCode/support/input_functions.py")
    print("5. Implement data caching")
    
    print("\nFILES TO FOCUS ON:")
    print("- SourceCode/support/input_functions.py (data loading)")
    print("- SourceCode/model_class.py (__init__ method)")
    
    print("\nPROFILE RESULTS LOCATION:")
    print("- Output/Profiles/backend_profile.prof")
    print("- Output/Profiles/model_run_profile.prof") 
    print("- Output/Profiles/memory_usage.png")

def main():
    """Main function"""
    print_summary()
    create_parquet_converter()
    next_steps()

if __name__ == "__main__":
    main()

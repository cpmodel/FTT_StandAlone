#!/usr/bin/env python3
"""
Memory Profiler for FTT Stand Alone
===================================
This script monitors memory usage of the FTT model.
"""

import psutil
import time
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

class MemoryMonitor:
    """Monitor memory usage during FTT execution"""
    
    def __init__(self):
        self.memory_usage = []
        self.timestamps = []
        self.process = psutil.Process()
    
    def start_monitoring(self, interval=0.1):
        """Start memory monitoring"""
        self.start_time = time.time()
        self.memory_usage = []
        self.timestamps = []
        
    def record_memory(self):
        """Record current memory usage"""
        memory_mb = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
        current_time = time.time() - self.start_time
        
        self.memory_usage.append(memory_mb)
        self.timestamps.append(current_time)
        
        return memory_mb
    
    def plot_memory_usage(self, save_path=None):
        """Plot memory usage over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.memory_usage, 'b-', linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('FTT Model Memory Usage Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add peak memory annotation
        max_memory = max(self.memory_usage)
        max_time = self.timestamps[self.memory_usage.index(max_memory)]
        plt.annotate(f'Peak: {max_memory:.1f} MB', 
                    xy=(max_time, max_memory), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Memory plot saved to: {save_path}")
        
        plt.show()
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        if not self.memory_usage:
            return {}
        
        return {
            'peak_memory_mb': max(self.memory_usage),
            'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage),
            'min_memory_mb': min(self.memory_usage),
            'total_time_seconds': max(self.timestamps) if self.timestamps else 0
        }

def profile_ftt_memory():
    """Profile memory usage of FTT components"""
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    
    print("Starting FTT memory profiling...")
    
    try:
        # Record initial memory
        initial_memory = monitor.record_memory()
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Import modules and record memory
        print("Importing ModelRun...")
        from SourceCode.model_class import ModelRun
        import_memory = monitor.record_memory()
        print(f"Memory after imports: {import_memory:.1f} MB")
        
        # Create model instance
        print("Creating ModelRun instance...")
        model = ModelRun()
        model_memory = monitor.record_memory()
        print(f"Memory after model creation: {model_memory:.1f} MB")
        
        # Load data (this is usually memory intensive)
        print("Model data loaded, checking memory...")
        data_memory = monitor.record_memory()
        print(f"Memory after data loading: {data_memory:.1f} MB")
        
        # Simulate some processing
        print("Simulating processing...")
        time.sleep(2)  # Simulate work
        final_memory = monitor.record_memory()
        print(f"Final memory usage: {final_memory:.1f} MB")
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        monitor.record_memory()
    
    # Generate report
    stats = monitor.get_memory_stats()
    print("\n" + "="*50)
    print("MEMORY USAGE SUMMARY")
    print("="*50)
    for key, value in stats.items():
        if 'memory' in key:
            print(f"{key.replace('_', ' ').title()}: {value:.1f} MB")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
    
    # Save plot
    output_dir = Path("./Output/Profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "memory_usage.png"
    monitor.plot_memory_usage(plot_path)
    
    return monitor

if __name__ == "__main__":
    profile_ftt_memory()

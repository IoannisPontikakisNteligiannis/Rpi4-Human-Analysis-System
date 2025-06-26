import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class FPSVisualizer:
    """Visualizer specifically for FPS data from recorded JSON files"""
    
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = None
        self.fps_data = []
        self.frame_times = []
        self.frame_types = []
        
    def load_data(self):
        """Load FPS data from JSON file"""
        try:
            with open(self.json_file_path, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} frames from {self.json_file_path}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.json_file_path} not found")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file {self.json_file_path}")
            return False
    
    def extract_fps_data(self):
        """Extract FPS data from frames - FIXED to handle both old and new formats"""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        self.fps_data = []
        self.frame_times = []
        self.frame_types = []
        
        for frame in self.data:
            fps_value = None
            timestamp = None
            frame_type = frame.get('frame_type', 'unknown')
            
            # FIXED: Handle both old format (nested in 'angles') and new format
            if 'fps' in frame:
                fps_value = frame['fps']
            elif 'angles' in frame and 'fps' in frame['angles']:
                fps_value = frame['angles']['fps']
            
            # Get timestamp
            if 'timestamp' in frame:
                timestamp = frame['timestamp']
            elif 'frame_number' in frame:
                # Fallback: estimate timestamp from frame number
                timestamp = frame['frame_number'] / 30.0  # Assume 30 FPS
            
            # Only add if we have valid data
            if fps_value is not None and timestamp is not None:
                # FIXED: Filter out unrealistic FPS values that could cause weird graphs
                if 0.1 <= fps_value <= 200:  # Reasonable FPS range
                    self.fps_data.append(fps_value)
                    self.frame_times.append(timestamp)
                    self.frame_types.append(frame_type)
        
        print(f"Extracted {len(self.fps_data)} valid FPS data points")
        
        if len(self.fps_data) == 0:
            print("ERROR: No valid FPS data found!")
            print("Sample frame structure:")
            if self.data:
                print(json.dumps(self.data[0], indent=2))
            return False
        
        return True
    
    def plot_fps(self, save_plot=True):
        """Create and display FPS plots"""
        if not self.fps_data:
            print("No FPS data available. Call extract_fps_data() first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'FPS Performance Analysis - {Path(self.json_file_path).name}', fontsize=16)
        
        # Plot 1: FPS over time with frame type coloring
        processing_mask = np.array(self.frame_types) == 'processing'
        skipped_mask = np.array(self.frame_types) == 'skipped'
        
        if np.any(processing_mask):
            processing_times = np.array(self.frame_times)[processing_mask]
            processing_fps = np.array(self.fps_data)[processing_mask]
            axes[0, 0].scatter(processing_times, processing_fps, 
                             c='blue', alpha=0.6, s=1, label='Processing frames')
        
        if np.any(skipped_mask):
            skipped_times = np.array(self.frame_times)[skipped_mask]
            skipped_fps = np.array(self.fps_data)[skipped_mask]
            axes[0, 0].scatter(skipped_times, skipped_fps, 
                             c='red', alpha=0.6, s=1, label='Skipped frames')
        
        # Add overall trend line
        axes[0, 0].plot(self.frame_times, self.fps_data, 'gray', alpha=0.3, linewidth=0.5)
        
        axes[0, 0].set_title('FPS Over Time')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Add moving average
        if len(self.fps_data) > 10:
            window_size = min(30, len(self.fps_data) // 10)
            moving_avg = np.convolve(self.fps_data, np.ones(window_size)/window_size, mode='valid')
            moving_avg_times = self.frame_times[window_size-1:]
            axes[0, 0].plot(moving_avg_times, moving_avg, 'darkgreen', linewidth=2, 
                           label=f'Moving Average ({window_size} frames)')
            axes[0, 0].legend()
        
        # Plot 2: FPS histogram
        axes[0, 1].hist(self.fps_data, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('FPS Distribution')
        axes[0, 1].set_xlabel('FPS')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(self.fps_data), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(self.fps_data):.1f} FPS')
        
        
        # Plot 3: Frame time analysis (more stable than FPS for visualization)
        frame_times_ms = [1000/fps for fps in self.fps_data]  # Convert to milliseconds
        
        axes[1, 0].plot(self.frame_times, frame_times_ms, 'purple', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Frame Time (ms) Over Time')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Frame Time (ms)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add target frame time lines
        axes[1, 0].axhline(1000/30, color='orange', linestyle='--', alpha=0.7, label='30 FPS target')
        axes[1, 0].legend()
        
        # Plot 4: Performance metrics
        axes[1, 1].axis('off')
        
        # Calculate statistics
        mean_fps = np.mean(self.fps_data)
        min_fps = np.min(self.fps_data)
        max_fps = np.max(self.fps_data)
        std_fps = np.std(self.fps_data)
        percentile_95 = np.percentile(self.fps_data, 95)
        percentile_5 = np.percentile(self.fps_data, 5)
        
        # Count frame types
        processing_count = sum(1 for ft in self.frame_types if ft == 'processing')
        skipped_count = sum(1 for ft in self.frame_types if ft == 'skipped')
        
        stats_text = f"""
Performance Statistics:

Mean FPS: {mean_fps:.1f}
Median FPS: {np.median(self.fps_data):.1f}
Min FPS: {min_fps:.1f}
Max FPS: {max_fps:.1f}
Std Dev: {std_fps:.1f}

Percentiles:
95th: {percentile_95:.1f} FPS
5th: {percentile_5:.1f} FPS

Frame Types:
Processing: {processing_count} ({processing_count/len(self.fps_data)*100:.1f}%)
Skipped: {skipped_count} ({skipped_count/len(self.fps_data)*100:.1f}%)

Total Frames: {len(self.fps_data)}
Duration: {self.frame_times[-1]:.1f}s
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            output_path = Path(self.json_file_path).parent / f"fps_analysis_{Path(self.json_file_path).stem}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"FPS analysis plot saved to {output_path}")
        
        plt.show()
    
    def print_fps_summary(self):
        """Print detailed FPS statistics"""
        if not self.fps_data:
            print("No FPS data available. Call extract_fps_data() first.")
            return
        
        print("\n=== FPS PERFORMANCE SUMMARY ===")
        print(f"Total frames analyzed: {len(self.fps_data)}")
        print(f"Session duration: {self.frame_times[-1]:.1f} seconds")
        print(f"Mean FPS: {np.mean(self.fps_data):.2f}")
        print(f"Median FPS: {np.median(self.fps_data):.2f}")
        print(f"Min FPS: {np.min(self.fps_data):.2f}")
        print(f"Max FPS: {np.max(self.fps_data):.2f}")
        print(f"Standard deviation: {np.std(self.fps_data):.2f}")
        
        # Percentiles
        print(f"\nPercentiles:")
        for p in [5, 25, 50, 75, 95]:
            print(f"  {p}th percentile: {np.percentile(self.fps_data, p):.2f} FPS")
        
        # Frame type breakdown
        processing_count = sum(1 for ft in self.frame_types if ft == 'processing')
        skipped_count = sum(1 for ft in self.frame_types if ft == 'skipped')
        print(f"\nFrame type breakdown:")
        print(f"  Processing frames: {processing_count} ({processing_count/len(self.fps_data)*100:.1f}%)")
        print(f"  Skipped frames: {skipped_count} ({skipped_count/len(self.fps_data)*100:.1f}%)")
        
        # Performance analysis
        low_fps_count = sum(1 for fps in self.fps_data if fps < 20)
        good_fps_count = sum(1 for fps in self.fps_data if fps >= 30)
        
        print(f"\nPerformance breakdown:")
        print(f"  Frames below 20 FPS: {low_fps_count} ({low_fps_count/len(self.fps_data)*100:.1f}%)")
        print(f"  Frames at 30+ FPS: {good_fps_count} ({good_fps_count/len(self.fps_data)*100:.1f}%)")

def main():
    """Main function to run the FPS visualizer"""
    
    # Look for FPS JSON files
    pose_dir = Path("exported_poses")
    
    if not pose_dir.exists():
        print(f"Error: Directory '{pose_dir}' doesn't exist!")
        return
    
    # Look for FPS JSON files
    fps_files = list(pose_dir.glob("fps_recording_*.json"))
    
    if not fps_files:
        print(f"No FPS recording JSON files found in '{pose_dir}'")
        print("Available files in directory:")
        all_files = list(pose_dir.iterdir())
        if all_files:
            for file in all_files:
                print(f"  - {file.name}")
        return
    
    # Show available files and use the most recent one
    print("Found FPS files:")
    for i, file in enumerate(fps_files):
        print(f"  {i+1}. {file.name}")
    
    # Use the most recent file
    fps_file = max(fps_files, key=lambda x: x.stat().st_mtime)
    print(f"\nUsing most recent file: {fps_file.name}")
    
    # Create visualizer
    visualizer = FPSVisualizer(fps_file)
    
    # Load and process data
    if visualizer.load_data():
        if visualizer.extract_fps_data():
            visualizer.print_fps_summary()
            visualizer.plot_fps(save_plot=True)
        else:
            print("Failed to extract valid FPS data - check file format")
    else:
        print("Failed to load data - check file format")

if __name__ == "__main__":
    main()
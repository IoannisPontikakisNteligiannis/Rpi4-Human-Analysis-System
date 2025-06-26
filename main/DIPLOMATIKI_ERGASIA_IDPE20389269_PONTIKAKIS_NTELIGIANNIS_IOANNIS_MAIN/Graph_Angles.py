import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class PoseAngleVisualizer:
    """visualizer for pose angles from recorded JSON data"""
    
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = None
        self.angles_data = {}
        
    def load_data(self):
        """Load pose data from JSON file"""
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
    
    def extract_angles(self):
        """Extract angle data from frames"""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        # Get all unique angle names
        angle_names = set()
        for frame in self.data:
            if 'angles' in frame:
                angle_names.update(frame['angles'].keys())
        
        # Initialize angle data storage
        for angle_name in angle_names:
            self.angles_data[angle_name] = []
        
        # Extract angle values for each frame
        for frame in self.data:
            if 'angles' in frame:
                for angle_name in angle_names:
                    value = frame['angles'].get(angle_name, None)
                    self.angles_data[angle_name].append(value)
            else:
                # If no angles in this frame, append None
                for angle_name in angle_names:
                    self.angles_data[angle_name].append(None)
        
        print(f"Extracted angles: {list(angle_names)}")
    
    def plot_angles(self, save_plot=True):
        """Create and display plots for all angles"""
        if not self.angles_data:
            print("No angle data available. Call extract_angles() first.")
            return
        
        # Create time axis (frame numbers)
        frames = list(range(len(self.data)))
        
        # Determine number of subplots needed
        num_angles = len(self.angles_data)
        if num_angles == 0:
            print("No angles found in the data")
            return
        
        # Calculate subplot layout
        cols = min(3, num_angles)  # Max 3 columns
        rows = (num_angles + cols - 1) // cols
        
        # Create figure with more space - KEY FIX #1
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5*rows))
        fig.suptitle('Pose Angles Over Time', fontsize=16, y=0.98)  # Move title up
        
        # Handle single subplot case
        if num_angles == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each angle
        for i, (angle_name, values) in enumerate(self.angles_data.items()):
            ax = axes[i]
            
            # Filter out None values for plotting
            valid_frames = []
            valid_values = []
            for frame, value in zip(frames, values):
                if value is not None:
                    valid_frames.append(frame)
                    valid_values.append(value)
            
            if valid_values:
                ax.plot(valid_frames, valid_values, linewidth=2, marker='o', markersize=2)
                
                # Clean up title formatting - KEY FIX #2
                ax.set_title(f'{angle_name}', fontsize=12, pad=18)
                
                # Better label spacing - KEY FIX #3
                ax.set_xlabel('Frame', fontsize=8)
                ax.set_ylabel('Angle (degrees)', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add statistics with better positioning - KEY FIX #4
                mean_val = np.mean(valid_values)
                ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7)
                
                # Position legend in upper right to avoid overlap - KEY FIX #5
                ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}°', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                # Improve tick spacing - KEY FIX #6
                ax.tick_params(axis='both', which='major', labelsize=9)
                
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{angle_name} (No Data)', fontsize=12, pad=15)
        
        # Hide unused subplots
        for i in range(num_angles, len(axes)):
            axes[i].set_visible(False)
        
        # Better spacing - KEY FIX #7
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        plt.subplots_adjust(hspace=0.6, wspace=0.3)  # Add more space between plots
        
        # Save plot if requested
        if save_plot:
            output_path = Path(self.json_file_path).parent / f"angles_plot_{Path(self.json_file_path).stem}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        
        plt.show()
    
    def print_angle_summary(self):
        """Print summary statistics for each angle"""
        if not self.angles_data:
            print("No angle data available. Call extract_angles() first.")
            return
        
        print("\n=== ANGLE SUMMARY ===")
        for angle_name, values in self.angles_data.items():
            # Filter out None values
            valid_values = [v for v in values if v is not None]
            
            if valid_values:
                print(f"\n{angle_name}:")
                print(f"  Frames with data: {len(valid_values)}/{len(values)}")
                print(f"  Mean: {np.mean(valid_values):.1f}°")
                print(f"  Min: {np.min(valid_values):.1f}°")
                print(f"  Max: {np.max(valid_values):.1f}°")
                print(f"  Std Dev: {np.std(valid_values):.1f}°")
            else:
                print(f"\n{angle_name}: No valid data")

def main():
    """Main function to run the visualizer"""
    
    pose_dir = Path("exported_poses")
    
    if not pose_dir.exists():
        print(f"Error: Directory '{pose_dir}' doesn't exist!")
        print("Make sure you've run your pose recorder first to create JSON files.")
        return
    
    # JSON files
    json_files = list(pose_dir.glob("pose_recording_*.json"))
    
    if not json_files:
        print(f"No pose recording JSON files found in '{pose_dir}'")
        print("Available files in directory:")
        all_files = list(pose_dir.iterdir())
        if all_files:
            for file in all_files:
                print(f"  - {file.name}")
        else:
            print("  (directory is empty)")
        return
    
    # Show available files and use the most recent one
    print("Found JSON files:")
    for i, file in enumerate(json_files):
        print(f"  {i+1}. {file.name}")
    
    # Use the most recent file
    json_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"\nUsing most recent file: {json_file.name}")
    
    # Create visualizer
    visualizer = PoseAngleVisualizer(json_file)
    
    # Load and process data
    if visualizer.load_data():
        visualizer.extract_angles()
        visualizer.print_angle_summary()
        visualizer.plot_angles(save_plot=True)
    else:
        print("Failed to load data - check file format")

if __name__ == "__main__":
    main()
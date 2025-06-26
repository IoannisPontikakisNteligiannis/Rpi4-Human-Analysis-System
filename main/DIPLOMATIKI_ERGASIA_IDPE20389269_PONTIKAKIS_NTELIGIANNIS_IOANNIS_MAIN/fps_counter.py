import time
import json
from pathlib import Path
from datetime import datetime
import os

class FPSCounter:
    """Handles FPS calculation with separate tracking for processing vs skipped frames"""
    
    def __init__(self, averaging_window=10):
        self.fps_values = []
        self.prev_frame_time = 0
        self.averaging_window = averaging_window
        
        # Separate tracking for processing vs skipped frames
        self.processing_fps_values = []
        self.skipped_fps_values = []
        self.processing_times = []
        self.skipped_times = []
        
        # For JSON export
        self.fps_history = []
        self.frame_count = 0
        self.session_start_time = time.time()
        
    def update(self, frame_type=None):
        """Update FPS calculation with optional frame type tracking"""
        curr_frame_time = time.time()
        dt = curr_frame_time - self.prev_frame_time
        
        # Initialize prev_frame_time on first call
        if self.prev_frame_time == 0:
            self.prev_frame_time = curr_frame_time
            return 0
        
        if dt > 0:
            fps = 1.0 / dt
            self.fps_values.append(fps)
            
            # Track by frame type if specified
            if frame_type == 'processing':
                self.processing_fps_values.append(fps)
                self.processing_times.append(dt)
                if len(self.processing_fps_values) > self.averaging_window:
                    self.processing_fps_values.pop(0)
                    self.processing_times.pop(0)
                    
            elif frame_type == 'skipped':
                self.skipped_fps_values.append(fps)
                self.skipped_times.append(dt)
                if len(self.skipped_fps_values) > self.averaging_window:
                    self.skipped_fps_values.pop(0)
                    self.skipped_times.pop(0)
            
            # Store FPS data properly
            self.fps_history.append({
                'frame': self.frame_count,
                'timestamp': curr_frame_time - self.session_start_time,
                'fps': fps,
                'dt': dt,
                'frame_time_ms': dt * 1000,
                'frame_type': frame_type or 'unknown'
            })
            self.frame_count += 1
            
            # Keep only the last N FPS values for averaging
            if len(self.fps_values) > self.averaging_window:
                self.fps_values.pop(0)
        
        self.prev_frame_time = curr_frame_time
        
        # Return average FPS
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0
    
    def get_current_fps(self):
        """Get the current average FPS without updating"""
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0
    
    def get_processing_fps(self):
        """Get average FPS for processing frames only"""
        return sum(self.processing_fps_values) / len(self.processing_fps_values) if self.processing_fps_values else 0
    
    def get_skipped_fps(self):
        """Get average FPS for skipped frames only"""
        return sum(self.skipped_fps_values) / len(self.skipped_fps_values) if self.skipped_fps_values else 0
    
    def get_processing_time_ms(self):
        """Get average processing time in milliseconds"""
        return (sum(self.processing_times) / len(self.processing_times) * 1000) if self.processing_times else 0
    
    def get_skipped_time_ms(self):
        """Get average skipped frame time in milliseconds"""
        return (sum(self.skipped_times) / len(self.skipped_times) * 1000) if self.skipped_times else 0
    
    def save_fps_data(self, filename=None):
        """Save FPS data to JSON file with better error handling"""
        if not self.fps_history:
            print("No FPS data to save")
            return None
            
        # Create export directory if it doesn't exist
        export_dir = Path("exported_poses")
        export_dir.mkdir(exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fps_recording_{timestamp}.json"
        
        filepath = export_dir / filename
        
        # Store FPS data in a cleaner format
        fps_data = []
        for entry in self.fps_history:
            fps_data.append({
                'frame_number': entry['frame'],
                'timestamp': entry['timestamp'],
                'frame_type': entry['frame_type'],
                'fps': entry['fps'],
                'frame_time_ms': entry['frame_time_ms'],
                'dt': entry['dt']
            })
        
        try:
            # Ensure we have write permissions
            with open(filepath, 'w') as f:
                json.dump(fps_data, f, indent=2)
            
            # Verify the file was written correctly
            if filepath.exists():
                file_size = filepath.stat().st_size
                print(f"FPS data saved successfully: {filepath.name} ({file_size} bytes)")
                return str(filepath)
            else:
                print(f"File save failed - file does not exist: {filepath}")
                return None
                
        except Exception as e:
            print(f"Error saving FPS data: {e}")
            return None
    
    @staticmethod
    def load_latest_fps_data():
        """Load the most recent FPS data file"""
        export_dir = Path("exported_poses")
        
        if not export_dir.exists():
            print(f"Export directory does not exist: {export_dir}")
            return None
        
        # Find all FPS recording files
        fps_files = list(export_dir.glob("fps_recording_*.json"))
        
        if not fps_files:
            print("No FPS recording files found")
            return None
        
        # Sort by filename (timestamp) - most recent first
        fps_files.sort(key=lambda f: f.name, reverse=True)
        latest_file = fps_files[0]
        
        print(f"Found {len(fps_files)} FPS files:")
        for i, f in enumerate(fps_files[:5]):  # Show first 5
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            size = f.stat().st_size
            marker = " <- LOADING" if i == 0 else ""
            print(f"  {f.name} - {size} bytes - {mtime}{marker}")
        
        if len(fps_files) > 5:
            print(f"  ... and {len(fps_files) - 5} more files")
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            print(f"Successfully loaded {len(data)} frames from {latest_file.name}")
            return data
            
        except Exception as e:
            print(f"Error loading {latest_file.name}: {e}")
            return None
    
    @staticmethod
    def load_specific_fps_file(filename):
        """Load a specific FPS data file"""
        export_dir = Path("exported_poses")
        filepath = export_dir / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            print(f"Available files in {export_dir}:")
            if export_dir.exists():
                for f in export_dir.glob("fps_recording_*.json"):
                    print(f"  {f.name}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print(f"Successfully loaded {len(data)} frames from {filename}")
            return data
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    @staticmethod
    def debug_file_system():
        """Debug file system issues"""
        print("=== File System Debug ===")
        print(f"Current working directory: {os.getcwd()}")
        
        export_dir = Path("exported_poses")
        print(f"Export directory path: {export_dir.absolute()}")
        print(f"Export directory exists: {export_dir.exists()}")
        
        if export_dir.exists():
            files = list(export_dir.glob("*.json"))
            print(f"Found {len(files)} JSON files:")
            
            for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                size = f.stat().st_size
                print(f"  {f.name} - {size} bytes - Modified: {mtime}")
        else:
            print("Export directory does not exist")
        
        print("=" * 30)
    
    def get_stats(self):
        """Get comprehensive FPS statistics"""
        if not self.fps_history:
            return {
                'total_frames': 0,
                'avg_fps': 0,
                'min_fps': 0,
                'max_fps': 0,
                'session_duration': 0,
                'processing_fps': 0,
                'skipped_fps': 0,
                'processing_time_ms': 0,
                'skipped_time_ms': 0
            }
        
        fps_values = [entry['fps'] for entry in self.fps_history]
        session_duration = time.time() - self.session_start_time
        
        return {
            'total_frames': len(self.fps_history),
            'avg_fps': sum(fps_values) / len(fps_values),
            'min_fps': min(fps_values),
            'max_fps': max(fps_values),
            'session_duration': session_duration,
            'processing_fps': self.get_processing_fps(),
            'skipped_fps': self.get_skipped_fps(),
            'processing_time_ms': self.get_processing_time_ms(),
            'skipped_time_ms': self.get_skipped_time_ms(),
            'processing_frames': len(self.processing_fps_values),
            'skipped_frames': len(self.skipped_fps_values)
        }
    
    def print_detailed_stats(self):
        """Print detailed performance statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("DETAILED FPS PERFORMANCE STATS")
        print("="*50)
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Session Duration: {stats['session_duration']:.1f}s")
        print(f"Overall Average FPS: {stats['avg_fps']:.1f}")
        print(f"Min/Max FPS: {stats['min_fps']:.1f} / {stats['max_fps']:.1f}")
        print("-" * 30)
        print(f"Processing Frames: {stats['processing_frames']}")
        print(f"Processing FPS: {stats['processing_fps']:.1f}")
        print(f"Processing Time: {stats['processing_time_ms']:.1f}ms")
        print("-" * 30)
        print(f"Skipped Frames: {stats['skipped_frames']}")
        print(f"Skipped FPS: {stats['skipped_fps']:.1f}")
        print(f"Skipped Time: {stats['skipped_time_ms']:.1f}ms")
        print("="*50)
    
    def reset(self):
        """Reset all FPS data"""
        self.fps_values = []
        self.fps_history = []
        self.processing_fps_values = []
        self.skipped_fps_values = []
        self.processing_times = []
        self.skipped_times = []
        self.frame_count = 0
        self.session_start_time = time.time()
        self.prev_frame_time = 0


# Example usage for debugging on Raspberry Pi
if __name__ == "__main__":
    # Debug the file system
    FPSCounter.debug_file_system()
    
    # Try to load the latest file
    print("\nAttempting to load latest FPS data...")
    data = FPSCounter.load_latest_fps_data()
    
    if data:
        print(f"Loaded data with {len(data)} frames")
        print(f"First frame: {data[0]}")
        print(f"Last frame: {data[-1]}")
    else:
        print("Failed to load data")
    
    # Try to load a specific file
    print("\nAttempting to load specific file...")
    specific_data = FPSCounter.load_specific_fps_file("fps_recording_20250623_235015.json")
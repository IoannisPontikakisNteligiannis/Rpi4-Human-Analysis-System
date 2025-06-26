import json
import datetime
from pathlib import Path

class PoseRecorder:
    """Handles recording and saving pose data"""
    
    def __init__(self):
        self.recording = False
        self.frames = []
        self.start_time = None
    
    def start_recording(self):
        """Start recording landmarks and angles"""
        self.recording = True
        self.frames = []
        self.start_time = datetime.datetime.now()
        print("Recording started...")
    
    def stop_recording(self):
        """Stop recording and save data"""
        if not self.recording:
            return None
        
        self.recording = False
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        print(f"Recording stopped. {len(self.frames)} frames recorded over {elapsed:.2f} seconds.")
        
        # Create directory if it doesn't exist
        output_dir = Path("exported_poses")
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"pose_recording_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.frames, f, indent=2)
        
        print(f"Recording saved to {filename}")
        return str(filename)
    
    def add_frame(self, landmarks, angles_dict):
        """Add a frame of landmark data with angles to the recording"""
        if not self.recording:
            return
        
        # Convert landmarks to list format for JSON serialization
        landmarks_list = []
        if landmarks:
            for landmark in landmarks.landmark:
                landmarks_list.append([
                    landmark.x, 
                    landmark.y, 
                    landmark.z, 
                    landmark.visibility
                ])
        
        frame_data = {
            "landmarks": landmarks_list,
            "angles": angles_dict
        }
        
        self.frames.append(frame_data)
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import os

class MediaPipeAnimator:
    def __init__(self, json_path):
        self.skeleton_connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (23, 25), (25, 27), (24, 26), (26, 28),
        ]
        self.load_data(json_path)
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.current_frame = 0
        self.key_points = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def load_data(self, json_path):
        json_dir = Path(json_path).parent if Path(json_path).is_file() else Path(json_path)
        
        # Filter for pose_recording files only
        pose_files = list(json_dir.glob("pose_recording*.json"))
        if not pose_files:
            raise FileNotFoundError("No pose_recording files found")
        
        # Get the most recent pose_recording file
        latest_pose_file = max(pose_files, key=os.path.getctime)
        print(f"Loading pose file: {latest_pose_file}")
        
        with open(latest_pose_file, 'r') as f:
            self.raw_data = json.load(f)
        
        self.frames = len(self.raw_data)
        self.keypoints = []

        print(f"Total frames in file: {self.frames}")

        for frame in self.raw_data:
            if 'landmarks' in frame:
                landmarks = np.array(frame['landmarks'])
                self.keypoints.append(landmarks)
            else:
                self.keypoints.append(np.zeros((33, 4)))

        self.keypoints = np.array(self.keypoints)
        print(f"Processed landmarks shape: {self.keypoints.shape}")

    def update_animation(self, frame):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(1, 0)
        visibility_threshold = 0.5

        try:
            landmarks = self.keypoints[frame]

            if landmarks is not None and landmarks.shape[1] == 4:
                for connection in self.skeleton_connections:
                    if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                        pt1, pt2 = landmarks[connection[0]], landmarks[connection[1]]
                        if pt1[3] > visibility_threshold and pt2[3] > visibility_threshold:
                            self.ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'green', linewidth=2, zorder=2)

                for i in self.key_points:
                    if i < len(landmarks) and landmarks[i][3] > visibility_threshold:
                        self.ax.scatter(landmarks[i][0], landmarks[i][1], c='red', s=50, zorder=3)

                if frame < len(self.raw_data) and 'angles' in self.raw_data[frame]:
                    angles = self.raw_data[frame]['angles']

                    joint_display_map = {
                                    'left_elbow': 13,
                                     'right_elbow': 14,
                                     'left_knee': 25,
                                    'right_knee': 26,
                                    'left_hip': 23,
                                    'right_hip': 24,
                                    'left_shoulder': 11,   
                                    'right_shoulder': 12    
                                }

                    for angle_name, landmark_index in joint_display_map.items():
                        angle_value = angles.get(angle_name)
                        if angle_value is not None and landmarks[landmark_index][3] > visibility_threshold:
                            x = landmarks[landmark_index][0]
                            y = landmarks[landmark_index][1]
                            offset_x = -0.03 if 'left' in angle_name else 0.01
                            offset_y = -0.02
                            self.ax.text(x + offset_x, y + offset_y,
                                         f"{angle_value:.1f}°", color='white', fontsize=8,
                                         bbox=dict(facecolor='blue', alpha=0.7))

                    y_pos = 0.05
                    for angle_name, value in angles.items():
                        if value is not None:
                            self.ax.text(0.05, y_pos, f"{angle_name}: {value:.1f}°",
                                         transform=self.ax.transAxes, color='white', fontsize=9,
                                         bbox=dict(facecolor='black', alpha=0.7))
                            y_pos += 0.05

            self.ax.set_title(f'Frame: {frame}/{self.frames-1}')
            self.ax.set_facecolor('black')

        except Exception as e:
            print(f"Error processing frame {frame}: {str(e)}")
            self.ax.set_title(f'Error in frame: {frame}')

    def create_animation(self, output_path='pose_animation.gif', fps=10):
        try:
            anim = FuncAnimation(self.fig, self.update_animation,
                                 frames=self.frames, interval=1000/fps)
            writer = PillowWriter(fps=fps)

            print("Starting animation save...")
            anim.save(output_path, writer=writer)
            plt.close()

        except Exception as e:
            print(f"Error creating animation: {str(e)}")
            plt.close()

def main():
    try:
        current_dir = Path.cwd()
        poses_dir = current_dir / "exported_poses"
        
        # Filter for pose_recording files only
        pose_files = list(poses_dir.glob("pose_recording*.json"))

        if not pose_files:
            print("No pose_recording JSON files found in exported_poses directory!")
            return

        # Get the most recent pose_recording file
        json_file = str(max(pose_files, key=lambda x: x.stat().st_mtime))
        print(f"Using recording: {json_file}")

        with open(json_file, 'r') as f:
            data = json.load(f)
            if len(data) > 0:
                print("First frame data structure:")
                print(json.dumps(data[0], indent=2))

        animator = MediaPipeAnimator(json_file)
        output_path = current_dir / "pose_animation_with_angles.gif"
        animator.create_animation(output_path=str(output_path))
        print(f"Animation saved to {output_path}")

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
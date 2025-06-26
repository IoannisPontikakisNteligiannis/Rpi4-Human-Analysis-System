# Rpi4-Human-Analysis-System [![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
A  computer vision system designed for Raspberry Pi 4 that performs real-time human pose analysis with specialized focus on exercise detection and biomechanical monitoring. The system combines MediaPipe pose estimation with custom exercise detection algorithms to provide comprehensive movement analysis and performance tracking.

* Real-time pose detection using MediaPipe
* Exercise recognition for bicep curls and shoulder abduction
* Bilateral arm tracking (left/right arm support)
* Joint angle calculation for 8 key body points
* Performance optimization for Raspberry Pi 4 (Frame Skipping)
* Data recording and analysis tools
* Interactive controls for live exercise switching
* FPS monitoring and performance analytics

# Controls
* "r" Start recording pose data
* "s" Stop recording and save data as json file
* "e" Toggle exercise detection on/off
* "a" arms (left ↔ right)
* "w" Switch exercises (bicep ↔ abduction)
* "t" Show statistics for current exercise
* "p" Print FPS statistic
* "f" Save FPS data to as json file
* "q" Quit application

# Configuration

Edit main.py to adjust:

FRAME_WIDTH = 640          # Camera resolution width <br/>
FRAME_HEIGHT = 480          # Camera resolution height <br/>
PROCESS_EVERY_N_FRAMES = 2  # Frame skipping (1=every frame, 2=every other) <br/>

# Exercise Sensitivity

Modify detector parameters in exercise detector files:

self.min_angle_threshold = 50   # Starting position angle <br/>
self.max_angle_threshold = 180  # Peak position angle <br/>
self.movement_threshold = 5     # Minimum movement to register <br/>

# Performance Issues

* Reduce resolution: Lower FRAME_WIDTH and FRAME_HEIGHT
* Increase frame skipping: Set PROCESS_EVERY_N_FRAMES = 3 or higher (detectors migh need adjusting the thresholds to work right)
* Disable visual effects: Comment out drawing functions in main loop

  # Performance with 2 frame skipping on Rpi4 8gb
  ![modelo0_askision_rpi4_new](https://github.com/user-attachments/assets/b013832b-cf74-4e6c-94d4-c8c49fc1ae3d)

  # Angle Performance on Neutral Pose![offset_gia_diplomatiki](https://github.com/user-attachments/assets/36fa0852-6d27-400a-9ba3-27b4869acc58)
  ![oudetero](https://github.com/user-attachments/assets/0991dd90-38c8-4b14-a349-23560f25a1cc)


  

  


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
![image](https://github.com/user-attachments/assets/48b09dd6-b4be-47dc-a988-4132d042daa0)


  # Angle Performance on Neutral Pose![offset_gia_diplomatiki](https://github.com/user-attachments/assets/36fa0852-6d27-400a-9ba3-27b4869acc58)
  ![oudetero](https://github.com/user-attachments/assets/0991dd90-38c8-4b14-a349-23560f25a1cc)

  
  # Angle Performance on Squats
  

  ![squats](https://github.com/user-attachments/assets/b505a383-6fb5-4e05-b099-bf2276e9721d)


  # Bicep Curls

![github_bicep_right](https://github.com/user-attachments/assets/83a096cb-41be-456a-ad6e-e3112df076df)
![github_bicep](https://github.com/user-attachments/assets/88492d63-c3ae-4e12-9feb-d65c009b3152)


# Shoulder Abduction

![left_Shoulder_github](https://github.com/user-attachments/assets/1b7eb6a8-5d95-4e13-a5b8-419111abf996)
![shoulder_github](https://github.com/user-attachments/assets/83700440-38d6-43b4-a836-d0b6a8c9937f)

# New version of the system is available here :
https://github.com/IoannisPontikakisNteligiannis/Human_pose_vol2




  

  


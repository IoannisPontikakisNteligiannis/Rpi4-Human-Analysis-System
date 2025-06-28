import cv2
import mediapipe as mp
import time

from pose_recorder import PoseRecorder
from angle_calculator import calculate_all_angles
from display_utils import DisplayManager
from fps_counter import FPSCounter
from bicep_curl_detector import BicepCurlDetector  
from shoulder_abduction_detector import ShoulderAbductionDetector

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    """Main function for recording pose data with PROPER frame skipping and FPS tracking"""
    # Configuration
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame for better performance
    
    # Set up video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Initialize components
    pose_recorder = PoseRecorder()
    display_manager = DisplayManager(FRAME_WIDTH, FRAME_HEIGHT)
    fps_counter = FPSCounter()
    
    # Initialize exercise detectors
    right_bicep_detector = BicepCurlDetector(arm='right')
    left_bicep_detector = BicepCurlDetector(arm='left')
    right_abduction_detector = ShoulderAbductionDetector(arm='right')
    left_abduction_detector = ShoulderAbductionDetector(arm='left')
    
    # Exercise selection variables
    current_exercise = 'bicep'
    current_detector = right_bicep_detector
    exercise_detection_enabled = True
    
    # Frame skipping variables - PROPER implementation
    frame_counter = 0
    saved_filename = None
    
    # Cache for skipped frames - store COMPLETE state
    cached_results = None
    cached_landmarks = None
    cached_angles_dict = None
    cached_exercise_results = None
    
    # Setup MediaPipe instance
    with mp_pose.Pose(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7, 
        model_complexity=0
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # FIXED: Frame skipping logic with proper FPS timing
            frame_counter += 1
            process_current_frame = (frame_counter % PROCESS_EVERY_N_FRAMES == 0)
            
            # Start timing for this frame
            frame_start_time = time.time()
            
            if process_current_frame:
                # === PROCESSING FRAME - Heavy computation ===
                
                # Recolor image to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Make detection - THIS IS THE EXPENSIVE OPERATION
                results = pose.process(image_rgb)
                
                # Initialize angles dictionary
                angles_dict = {
                    "left_elbow": None, "right_elbow": None, 
                    "left_shoulder": None, "right_shoulder": None,
                    "left_hip": None, "right_hip": None,
                    "left_knee": None, "right_knee": None
                }
                
                exercise_results = None
                
                if results and results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    # EXPENSIVE: Calculate all angles
                    angles_dict = calculate_all_angles(landmarks)
                    
                    # Exercise detection
                    if exercise_detection_enabled:
                        exercise_results = current_detector.update(angles_dict)
                    
                    # Add frame to recording if active (only on processing frames)
                    if pose_recorder.recording:
                        pose_recorder.add_frame(results.pose_landmarks, angles_dict)
                else:
                    # No landmarks detected
                    if exercise_detection_enabled:
                        exercise_results = current_detector.update(angles_dict)
                
                # Cache ALL results for skipped frames
                cached_results = results
                cached_landmarks = results.pose_landmarks.landmark if results and results.pose_landmarks else None
                cached_angles_dict = angles_dict.copy()
                cached_exercise_results = exercise_results
                
                # Update FPS counter AFTER processing is complete
                current_fps = fps_counter.update('processing')
                
            else:
                
                # Use cached data - NO heavy computation
                results = cached_results
                landmarks = cached_landmarks
                angles_dict = cached_angles_dict if cached_angles_dict else {
                    "left_elbow": None, "right_elbow": None, 
                    "left_shoulder": None, "right_shoulder": None,
                    "left_hip": None, "right_hip": None,
                    "left_knee": None, "right_knee": None
                }
                exercise_results = cached_exercise_results
                
                # Update FPS counter AFTER using cache
                current_fps = fps_counter.update('skipped')
        
            # Recolor back to BGR for display
            image = frame.copy()
            
            if results and results.pose_landmarks:
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Draw angle measurements
                display_manager.draw_all_angles(image, landmarks, angles_dict)
            else:
                # Draw angle measurements (will show "Unknown" for all)
                display_manager.draw_all_angles(image, None, angles_dict)
            
            # Draw exercise information if detection is enabled
            if exercise_detection_enabled and exercise_results:
                display_manager.draw_exercise_info(image, exercise_results)
            
            # Draw status information
            display_manager.draw_status_info(
                image, 
                current_fps, 
                pose_recorder.recording, 
                saved_filename,
                exercise_detection_enabled,
                current_detector.arm,
                current_exercise
            )
            
            # Display the frame
            cv2.imshow('MediaPipe Pose with Exercise Detection', image)
            
            # Handle keyboard input
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('r'):
                pose_recorder.start_recording()
                saved_filename = None
            
            elif key == ord('s'):
                if pose_recorder.recording:
                    saved_filename = pose_recorder.stop_recording()
            
            elif key == ord('e'):
                exercise_detection_enabled = not exercise_detection_enabled
                print(f"Exercise detection: {'ON' if exercise_detection_enabled else 'OFF'}")
            
            elif key == ord('a'):
                # Switch between left and right arm detection
                if current_exercise == 'bicep':
                    if current_detector == right_bicep_detector:
                        current_detector = left_bicep_detector
                        print("Switched to LEFT arm bicep detection")
                    else:
                        current_detector = right_bicep_detector
                        print("Switched to RIGHT arm bicep detection")
                else:  # abduction
                    if current_detector == right_abduction_detector:
                        current_detector = left_abduction_detector
                        print("Switched to LEFT arm abduction detection")
                    else:
                        current_detector = right_abduction_detector
                        print("Switched to RIGHT arm abduction detection")
            
            elif key == ord('w'): 
                # Switch between bicep curls and shoulder abduction
                if current_exercise == 'bicep':
                    current_exercise = 'abduction'
                    if current_detector.arm == 'right':
                        current_detector = right_abduction_detector
                    else:
                        current_detector = left_abduction_detector
                    print(f"Switched to SHOULDER ABDUCTION exercise ({current_detector.arm.upper()} arm)")
                else:
                    current_exercise = 'bicep'
                    if current_detector.arm == 'right':
                        current_detector = right_bicep_detector
                    else:
                        current_detector = left_bicep_detector
                    print(f"Switched to BICEP CURL exercise ({current_detector.arm.upper()} arm)")
            
            elif key == ord('x'):
                current_detector.reset()
                exercise_name = "Bicep Curl" if current_exercise == 'bicep' else "Elbow Abduction"
                print(f"Reset {current_detector.arm} arm {exercise_name} counter")
            
            elif key == ord('t'):
                stats = current_detector.get_stats()
                exercise_name = "BICEP CURL" if current_exercise == 'bicep' else "ELBOW ABDUCTION"
                print(f"\n== {current_detector.arm.upper()} ARM {exercise_name} STATS ===")
                print(f"Total Reps: {stats['total_reps']}")
                print(f"Average Duration: {stats['avg_duration']}s")
                print(f"Average Range of Motion: {stats['avg_range_of_motion']}")
                if stats['total_reps'] > 0:
                    print(f"Last Rep Duration: {stats['last_rep_duration']}s")
                print("=" * 50)
            
            elif key == ord('p'):
                fps_counter.print_detailed_stats()
            
            elif key == ord('f'):
                fps_filename = fps_counter.save_fps_data()
                if fps_filename:
                    print(f"FPS data saved: {fps_filename}")

            elif key == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

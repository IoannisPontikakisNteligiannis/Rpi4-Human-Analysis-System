import cv2
import mediapipe as mp

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
    """Main function for recording pose data with angles and exercise detection"""
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
    
    # Initialize exercise detectors for exercises
    right_bicep_detector = BicepCurlDetector(arm='right')
    left_bicep_detector = BicepCurlDetector(arm='left')
    right_abduction_detector = ShoulderAbductionDetector(arm='right')
    left_abduction_detector = ShoulderAbductionDetector(arm='left')
    
    # Exercise selection variables
    current_exercise = 'bicep'  # 'bicep' or 'abduction'
    current_detector = right_bicep_detector  # Default detector
    exercise_detection_enabled = True
    
    # Frame skipping variables for performance optimization
    skip_frames = 0
    last_pose_result = None  # Store last pose detection result
    saved_filename = None  # For showing save confirmation
    
    # Setup MediaPipe instance
    with mp_pose.Pose(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7, 
        model_complexity=0  # Can be reduced to 0 for even better performance
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Frame skipping logic for performance optimization
            skip_frames += 1
            process_current_frame = (skip_frames % PROCESS_EVERY_N_FRAMES == 0)
            
            # Update FPS with frame type information
            if process_current_frame:
                current_fps = fps_counter.update('processing')
                skip_frames = 0  # Reset counter
                
                # Recolor image to RGB for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Make detection
                results = pose.process(image_rgb)
                last_pose_result = results  # Store result for skipped frames
            else:
                current_fps = fps_counter.update('skipped')
                # Use last pose detection result for skipped frames
                results = last_pose_result
            
            # Recolor back to BGR for display
            image = frame.copy()
            
            # Initialize angles dictionary with all angles
            angles_dict = {
                "left_elbow": None, "right_elbow": None, 
                "left_shoulder": None, "right_shoulder": None,
                "left_hip": None, "right_hip": None,
                "left_knee": None, "right_knee": None
            }
            
            # Initialize exercise results
            exercise_results = None
            
            if results and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                angles_dict = calculate_all_angles(landmarks)
                
                # Exercise detection
                if exercise_detection_enabled:
                    exercise_results = current_detector.update(angles_dict)
                
                # Add frame to recording if active
                if pose_recorder.recording:
                    pose_recorder.add_frame(results.pose_landmarks, angles_dict)
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
                
                # Draw angle measurements
                display_manager.draw_all_angles(image, landmarks, angles_dict)
            else:
                # No landmarks detected
                if exercise_detection_enabled:
                    exercise_results = current_detector.update(angles_dict)
                
                # Draw angle measurements (will show "Unknown" for all)
                display_manager.draw_all_angles(image, None, angles_dict)
            
            # Draw exercise information if detection is enabled
            if exercise_detection_enabled and exercise_results:
                display_manager.draw_exercise_info(image, exercise_results)
            
            # Draw status information with updated instructions
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
            
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('r'):
                pose_recorder.start_recording()
                saved_filename = None  # Clear any previous save message
            
            elif key == ord('s'):
                if pose_recorder.recording:
                    saved_filename = pose_recorder.stop_recording()
            
            elif key == ord('e'):
                # Toggle exercise detection
                exercise_detection_enabled = not exercise_detection_enabled
                print(f"Exercise detection: {'ON' if exercise_detection_enabled else 'OFF'}")
            
            elif key == ord('a'):
                # Switch between left and right arm detection (same exercise)
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
                    # Switch to corresponding abduction detector
                    if current_detector.arm == 'right':
                        current_detector = right_abduction_detector
                    else:
                        current_detector = left_abduction_detector
                    print(f"Switched to SHOULDER ABDUCTION exercise ({current_detector.arm.upper()} arm)")
                else:
                    current_exercise = 'bicep'
                    # Switch to corresponding bicep detector
                    if current_detector.arm == 'right':
                        current_detector = right_bicep_detector
                    else:
                        current_detector = left_bicep_detector
                    print(f"Switched to BICEP CURL exercise ({current_detector.arm.upper()} arm)")
            
            elif key == ord('x'):
                # Reset exercise counter
                current_detector.reset()
                exercise_name = "Bicep Curl" if current_exercise == 'bicep' else "Elbow Abduction"
                print(f"Reset {current_detector.arm} arm {exercise_name} counter")
            
            elif key == ord('t'):
                # Show exercise statistics
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
                # Print detailed FPS statistics
                fps_counter.print_detailed_stats()
            
            elif key == ord('f'):
                # Save FPS data
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
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

class DisplayManager:
    """Handles all display and visualization tasks including exercise information"""
    
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def draw_angle_text(self, image, landmarks, angle, angle_name, landmark_index, color=(255, 255, 255), offset=(0, 0)):
        """Draw angle text at landmark position with optional offset"""
        if angle is not None:
            landmark_pixel = tuple(np.multiply([
                landmarks[landmark_index].x,
                landmarks[landmark_index].y
            ], [self.frame_width, self.frame_height]).astype(int))
            
            # Apply offset
            landmark_pixel = (landmark_pixel[0] + offset[0], landmark_pixel[1] + offset[1])
            
            cv2.putText(image, f"{angle_name}: {angle:.1f}", 
                       landmark_pixel, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_8)
        else:
            # Show "Unknown" if angle can't be calculated
            if landmarks[landmark_index].visibility > 0.3:
                landmark_pixel = tuple(np.multiply([
                    landmarks[landmark_index].x,
                    landmarks[landmark_index].y
                ], [self.frame_width, self.frame_height]).astype(int))
                
                # Apply offset
                landmark_pixel = (landmark_pixel[0] + offset[0], landmark_pixel[1] + offset[1])
                
                cv2.putText(image, f"{angle_name}: Unknown", 
                           landmark_pixel, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
    
    def draw_all_angles(self, image, landmarks, angles_dict):
        """Draw all angle measurements on the image"""
        if not landmarks:
            # No landmarks detected - show all as unknown
            positions = [(50, 100), (50, 130), (50, 160), (50, 190), (50, 220), (50, 250), (50, 280), (50, 310)]
            labels = ["LE: Unknown", "RE: Unknown", "LS: Unknown", "RS: Unknown", 
                     "LH: Unknown", "RH: Unknown", "LK: Unknown", "RK: Unknown"]
            
            for pos, label in zip(positions, labels):
                cv2.putText(image, label, pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
            return
        
        # Draw left elbow angle
        if angles_dict["left_elbow"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["left_elbow"], 
                               "LE", mp_pose.PoseLandmark.LEFT_ELBOW.value)
        else:
            self.draw_angle_text(image, landmarks, None, 
                               "LE", mp_pose.PoseLandmark.LEFT_ELBOW.value)
        
        # Draw right elbow angle
        if angles_dict["right_elbow"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["right_elbow"], 
                               "RE", mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        else:
            self.draw_angle_text(image, landmarks, None, 
                               "RE", mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        
        # Draw left shoulder angle (offset up from shoulder)
        if angles_dict["left_shoulder"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["left_shoulder"], 
                               "LS", mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                               offset=(0, -15))
        else:
            if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "LS", mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                                   offset=(0, -15))
            else:
                cv2.putText(image, "LS: Unknown", (50, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw right shoulder angle (offset up from shoulder)
        if angles_dict["right_shoulder"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["right_shoulder"], 
                               "RS", mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                               offset=(0, -15))
        else:
            if landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "RS", mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                                   offset=(0, -15))
            else:
                cv2.putText(image, "RS: Unknown", (50, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw left hip angle
        if angles_dict["left_hip"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["left_hip"], 
                               "LH", mp_pose.PoseLandmark.LEFT_HIP.value)
        else:
            if landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "LH", mp_pose.PoseLandmark.LEFT_HIP.value)
            else:
                cv2.putText(image, "LH: Unknown", (50, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw right hip angle
        if angles_dict["right_hip"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["right_hip"], 
                               "RH", mp_pose.PoseLandmark.RIGHT_HIP.value)
        else:
            if landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "RH", mp_pose.PoseLandmark.RIGHT_HIP.value)
            else:
                cv2.putText(image, "RH: Unknown", (50, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw left knee angle
        if angles_dict["left_knee"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["left_knee"], 
                               "LK", mp_pose.PoseLandmark.LEFT_KNEE.value)
        else:
            if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "LK", mp_pose.PoseLandmark.LEFT_KNEE.value)
            else:
                cv2.putText(image, "LK: Unknown", (50, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
        
        # Draw right knee angle
        if angles_dict["right_knee"] is not None:
            self.draw_angle_text(image, landmarks, angles_dict["right_knee"], 
                               "RK", mp_pose.PoseLandmark.RIGHT_KNEE.value)
        else:
            if landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.3:
                self.draw_angle_text(image, landmarks, None, 
                                   "RK", mp_pose.PoseLandmark.RIGHT_KNEE.value)
            else:
                cv2.putText(image, "RK: Unknown", (50, 310), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_8)
    
    def draw_exercise_info(self, image, exercise_results):
        """Draw exercise detection information"""
        if not exercise_results:
            return
        
        # Exercise info panel background (semi-transparent)
        overlay = image.copy()
        cv2.rectangle(overlay, (350, 50), (630, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Exercise info panel border 
        border_color = (255, 255, 255)  # Default white
        if 'Abduction' in exercise_results.get('exercise', ''):
            border_color = (255, 255, 255)  
        elif 'Bicep' in exercise_results.get('exercise', ''):
            border_color = (255, 255, 255) 
        
        cv2.rectangle(image, (350, 50), (630, 300), border_color, 2)
        
        # Title with exercise-specific color
        title_color = (255, 255, 255) if 'Abduction' in exercise_results.get('exercise', '') else (255, 255, 255)
        cv2.putText(image, "EXERCISE TRACKER", (360, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2, cv2.LINE_8)
        
        y_pos = 110
        
        # Exercise name and arm
        exercise_text = f"{exercise_results['exercise']} ({exercise_results['arm']})"
        cv2.putText(image, exercise_text, (360, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_8)
        y_pos += 30
        
        # Rep count 
        rep_text = f"REPS: {exercise_results['rep_count']}"
        cv2.putText(image, rep_text, (360, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_8)
        y_pos += 35
        
        # Current state
        state_text = f"State: {exercise_results['state']}"
        state_color = self._get_state_color(exercise_results['state'])
        cv2.putText(image, state_text, (360, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2, cv2.LINE_8)
        y_pos += 25
        
        # Current angle
        if exercise_results['angle'] is not None:
            angle_text = f"Angle: {exercise_results['angle']}"
            cv2.putText(image, angle_text, (360, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_8)
        else:
            cv2.putText(image, "Angle: N/A", (360, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_8)
        y_pos += 25
        
        # Peak and valley angles (if available)
        if exercise_results.get('peak_angle') is not None:
            peak_text = f"Peak: {exercise_results['peak_angle']}"
            cv2.putText(image, peak_text, (360, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_8)
            y_pos += 20
            
        if exercise_results.get('valley_angle') is not None:
            valley_text = f"Valley: {exercise_results['valley_angle']}"
            cv2.putText(image, valley_text, (360, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_8)
            y_pos += 20            

       
        # Feedback (word-wrapped if too long)
        feedback = exercise_results.get('feedback', '')
        if feedback:
            self._draw_wrapped_text(image, feedback, (360, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _get_state_color(self, state):
        """Get color for exercise state"""
        state_colors = {
            'Neutral': (255, 255, 255),
            'Up Phase': (0, 255, 0),
            'Down Phase': (255, 0, 0),
            'No Detection': (0, 0, 255)
        }
        return state_colors.get(state, (255, 255, 255))
    
    def _draw_wrapped_text(self, image, text, position, font, scale, color, thickness):
        """Draw text with word wrapping"""
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
            
            if text_width < 250:  # Max width for text
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        x, y = position
        for i, line in enumerate(lines):
            cv2.putText(image, line, (x, y + i * 20), font, scale, color, thickness, cv2.LINE_8)
    
    def draw_status_info(self, image, fps, is_recording, filename=None, exercise_enabled=True, current_arm='right', current_exercise='bicep'):
        """Draw FPS, recording status, and instructions"""
        # Display FPS
        cv2.putText(image, f"FPS: {fps:.1f}", (500, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_8)
        
        # Display recording status
        if is_recording:
            cv2.putText(image, "RECORDING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_8)
        
        # Display exercise detection status with exercise type
        exercise_name = "Bicep Curl" if current_exercise == 'bicep' else "Shoulder Abduction"
        exercise_status = f"Exercise: {exercise_name} {'ON' if exercise_enabled else 'OFF'} ({current_arm.upper()})"
        cv2.putText(image, exercise_status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_8)
        
        # Display instructions 
        instructions = [
            "Press 'r' to start recording",
            "Press 's' to stop recording and save",
            "Press 'e' to toggle exercise detection",
            "Press 'a' to switch arm (L/R)",
            "Press 'w' to switch exercise type",  
            "Press 'x' to reset exercise counter",
            "Press 't' to show exercise stats",
            "Press 'q' to quit"
        ]
        
        y_start = 340
        for i, instruction in enumerate(instructions):
            y_pos = y_start + i * 18
             # Draw outline by drawing black text at multiple offset positions
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                cv2.putText(image, instruction, (10 + dx, y_pos + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_8)
             # Draw main white text
            cv2.putText(image, instruction, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_8)
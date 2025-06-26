import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_angle(point1, point2, point3):
    """
    Calculate the angle at point2 (point1-point2-point3).
    Returns angle in 0-180°.
    
    Parameters:
        point1 (list): First point [x, y]
        point2 (list): Middle point (apex) [x, y]
        point3 (list): Third point [x, y]
    
    Returns:
        float: Angle in degrees (0-180), or None if invalid
    """
    try:
        # Check if any of the landmarks has low visibility or invalid position
        if None in (point1, point2, point3):
            return None
            
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)
        
        # Calculate vectors
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Check for zero distance (landmarks too close or improperly detected)
        vector1_mag = np.linalg.norm(vector1)
        vector2_mag = np.linalg.norm(vector2)
        
        # If vectors are too small, landmarks are likely misdetected
        MIN_VECTOR_LENGTH = 0.03 # Minimum sensible distance between landmarks 
        if vector1_mag < MIN_VECTOR_LENGTH or vector2_mag < MIN_VECTOR_LENGTH:
            return None
        
        # Dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        
        # Cosine of the angle
        cos_angle = dot_product / (vector1_mag * vector2_mag)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        
        # Angle in degrees (0-180°)
        angle = np.arccos(cos_angle) * 180.0 / np.pi     
        return angle
    except (TypeError, ValueError) as e:
        print(f"Error calculating angle: {e}")
        return None

def check_landmark_visibility(landmarks, landmark_indices, min_visibility=0.7):
    """
    Check if all landmarks in the specified indices have sufficient visibility.
    
    Parameters:
        landmarks: MediaPipe landmarks object
        landmark_indices: List of landmark indices to check
        min_visibility: Minimum visibility threshold (0-1)
        
    Returns:
        bool: True if all landmarks have sufficient visibility, False otherwise
    """
    if not landmarks:
        return False
        
    for idx in landmark_indices:
        if landmarks[idx].visibility < min_visibility:
            return False
    
    return True

def calculate_all_angles(landmarks):
    """
    Calculate all joint angles from MediaPipe landmarks.
    
    Parameters:
        landmarks: MediaPipe landmarks object
        
    Returns:
        dict: Dictionary containing all calculated angles
    """
    angles_dict = {
        "left_elbow": None,
        "right_elbow": None,
        "left_shoulder": None,
        "right_shoulder": None,
        "left_hip": None,
        "right_hip": None,
        "left_knee": None,
        "right_knee": None
    }
    
    if not landmarks:
        return angles_dict
    
    # Check visibility for different body parts
    left_arm_visible = check_landmark_visibility(
        landmarks, 
        [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
         mp_pose.PoseLandmark.LEFT_ELBOW.value,
         mp_pose.PoseLandmark.LEFT_WRIST.value],
        min_visibility=0.7
    )
    
    right_arm_visible = check_landmark_visibility(
        landmarks, 
        [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
         mp_pose.PoseLandmark.RIGHT_ELBOW.value,
         mp_pose.PoseLandmark.RIGHT_WRIST.value],
        min_visibility=0.7
    )
    
    left_shoulder_visible = check_landmark_visibility(
        landmarks, 
        [mp_pose.PoseLandmark.LEFT_HIP.value,
         mp_pose.PoseLandmark.LEFT_SHOULDER.value,
         mp_pose.PoseLandmark.LEFT_ELBOW.value],
        min_visibility=0.7
    )
    
    right_shoulder_visible = check_landmark_visibility(
        landmarks, 
        [mp_pose.PoseLandmark.RIGHT_HIP.value,
         mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
         mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        min_visibility=0.7
    )
    
    # Check visibility for hip angles
    left_hip_visible = check_landmark_visibility(
        landmarks,
        [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
         mp_pose.PoseLandmark.LEFT_HIP.value,
         mp_pose.PoseLandmark.LEFT_KNEE.value],
        min_visibility=0.7
    )
    
    right_hip_visible = check_landmark_visibility(
        landmarks,
        [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
         mp_pose.PoseLandmark.RIGHT_HIP.value,
         mp_pose.PoseLandmark.RIGHT_KNEE.value],
        min_visibility=0.7
    )
    
    # Check visibility for knee angles
    left_knee_visible = check_landmark_visibility(
        landmarks,
        [mp_pose.PoseLandmark.LEFT_HIP.value,
         mp_pose.PoseLandmark.LEFT_KNEE.value,
         mp_pose.PoseLandmark.LEFT_ANKLE.value],
        min_visibility=0.7
    )
    
    right_knee_visible = check_landmark_visibility(
        landmarks,
        [mp_pose.PoseLandmark.RIGHT_HIP.value,
         mp_pose.PoseLandmark.RIGHT_KNEE.value,
         mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        min_visibility=0.7
    )
    
    # Calculate LEFT elbow angle if visible
    if left_arm_visible:
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        vector_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        # Convert to anatomical angle (0° = full extension, increases with flexion)
        angles_dict["left_elbow"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate RIGHT elbow angle if visible
    if right_arm_visible:
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        vector_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        # Convert to anatomical angle (0° = full extension, increases with flexion)
        angles_dict["right_elbow"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate LEFT shoulder angle if visible
    if left_shoulder_visible:
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        
        angles_dict["left_shoulder"] = calculate_angle(left_hip, left_shoulder, left_elbow)
    
    # Calculate RIGHT shoulder angle if visible
    if right_shoulder_visible:
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        
        angles_dict["right_shoulder"] = calculate_angle(right_hip, right_shoulder, right_elbow)
    
    # Calculate LEFT hip angle if visible
    if left_hip_visible:
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        
        vector_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        # Convert to anatomical angle (0° = straight leg, increases with flexion)
        angles_dict["left_hip"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate RIGHT hip angle if visible
    if right_hip_visible:
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        
        vector_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        # Convert to anatomical angle (0° = straight leg, increases with flexion)
        angles_dict["right_hip"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate LEFT knee angle if visible
    if left_knee_visible:
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        vector_angle = calculate_angle(left_hip, left_knee, left_ankle)
        # Convert to anatomical angle (0° = full extension, increases with flexion)
        angles_dict["left_knee"] = 180 - vector_angle if vector_angle is not None else None
    
    # Calculate RIGHT knee angle if visible
    if right_knee_visible:
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        vector_angle = calculate_angle(right_hip, right_knee, right_ankle)
        # Convert to anatomical angle (0° = full extension, increases with flexion)
        angles_dict["right_knee"] = 180 - vector_angle if vector_angle is not None else None
    
    return angles_dict
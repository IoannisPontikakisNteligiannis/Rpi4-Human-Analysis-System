import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List

class ExerciseState(Enum):
    """States for exercise detection"""
    NEUTRAL = "neutral"
    UP_PHASE = "up_phase"  # Lifting arm up (abduction)
    DOWN_PHASE = "down_phase"  # Lowering arm down (adduction)

@dataclass
class RepData:
    """Data structure to store information about a single repetition"""
    start_time: float
    end_time: Optional[float] = None
    max_angle: Optional[float] = None
    min_angle: Optional[float] = None
    duration: Optional[float] = None
    
    def complete_rep(self, end_time: float):
        """Mark repetition as complete"""
        self.end_time = end_time
        if self.start_time:
            self.duration = end_time - self.start_time

class ShoulderAbductionDetector:
    
    def __init__(self, arm='right'):
        """
        Initialize shoulder abduction detector for shoulder exercises
        
        Parameters:
            arm (str): Which arm to track ('left' or 'right') - defaults to 'right'
        """
        # Ensure arm is either 'left' or 'right', default to 'right'
        if arm.lower() not in ['left', 'right']:
            print(f"Warning: Invalid arm '{arm}' specified. Defaulting to 'right'.")
            arm = 'right'
        
        self.arm = arm.lower()
        self.angle_key = f"{self.arm}_shoulder"  # Using shoulder angle for abduction
        
        # Thresholds for shoulder abduction (arm lifting to the side)
        self.min_angle_threshold = 20   # Arms at side (rest position)
        self.max_angle_threshold = 180  # Arms raised to shoulder level or higher
        self.movement_threshold = 8     # Minimum movement to register
        
        # Adaptive sustained movement tracking
        self.sustained_frames_fast = 2  # For fast movements
        self.sustained_frames_slow = 2  # For slow movements 
        self.up_movement_count = 0      # Count consecutive up movements
        self.down_movement_count = 0    # Count consecutive down movements
        self.slow_movement_threshold = 3 # Per-frame movement for slow detection
        
        # State tracking
        self.state = ExerciseState.NEUTRAL
        self.rep_count = 0
        self.current_rep = None
        self.rep_history: List[RepData] = []
        
        # Angle tracking
        self.angle_history = []
        self.history_size = 6       
        self.last_angle = None
        self.peak_angle = None
        self.valley_angle = None
        
        # Timing
        self.state_change_time = time.time()
        self.min_phase_duration = 0.2   # Minimum time in each phase 
        # Current movement tracking
        self.current_movement_direction = 0  # -1, 0, or 1
        self.movement_confirmation_threshold = 2.5  
        self.cumulative_movement = 0  # Track total movement over time
    
    def switch_arm(self, new_arm='right'):
        """
        Switch to tracking a different arm
        
        Parameters:
            new_arm (str): Which arm to track ('left' or 'right') - defaults to 'right'
        """
        # Validate arm parameter
        if new_arm.lower() not in ['left', 'right']:
            print(f"Warning: Invalid arm '{new_arm}' specified. Keeping current arm '{self.arm}'.")
            return False
        
        old_arm = self.arm
        self.arm = new_arm.lower()
        self.angle_key = f"{self.arm}_shoulder"
        
        # Reset the detector when switching arms
        self.reset()
        
        print(f"Switched from {old_arm} arm to {self.arm} arm detection")
        return True
    
    def get_current_arm(self):
        """Get the currently tracked arm"""
        return self.arm
        
    def _update_angle_history(self, angle: float):
        """Track recent angles"""
        self.angle_history.append(angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
    
    def _get_sustained_movement_direction(self) -> int:
        """
        Determine if there's sustained movement in one direction
        Returns: -1 (down), 0 (no clear direction), 1 (up)
        """
        if len(self.angle_history) < 3:
            return 0
        
        # Look at recent angle changes
        recent_changes = []
        for i in range(len(self.angle_history) - 3, len(self.angle_history)):
            if i > 0:
                change = self.angle_history[i] - self.angle_history[i-1]
                recent_changes.append(change)
        
        if not recent_changes:
            return 0
        
        # Calculate total movement over the window
        total_movement = self.angle_history[-1] - self.angle_history[-4] if len(self.angle_history) >= 4 else 0
        
        # Detect both fast and slow movements 
        fast_up_moves = sum(1 for change in recent_changes if change > 4) 
        fast_down_moves = sum(1 for change in recent_changes if change < -4)
        
        slow_up_moves = sum(1 for change in recent_changes if change > 0.3)  
        slow_down_moves = sum(1 for change in recent_changes if change < -0.3)
        
        # Update cumulative movement for slow detection
        if len(recent_changes) > 0:
            avg_change = sum(recent_changes) / len(recent_changes)
            if abs(avg_change) > 0.2:
                self.cumulative_movement += avg_change
            else:
                self.cumulative_movement *= 0.92  # Gradual decay
        
        # Fast movement detection
        if fast_up_moves >= 2 and total_movement > self.movement_confirmation_threshold:
            self.up_movement_count += 1
            self.down_movement_count = 0
            self.cumulative_movement = max(0, self.cumulative_movement)
            return 1
        elif fast_down_moves >= 2 and total_movement < -self.movement_confirmation_threshold:
            self.down_movement_count += 1
            self.up_movement_count = 0
            self.cumulative_movement = min(0, self.cumulative_movement)
            return -1
        
        # Slow movement detection
        elif slow_up_moves >= 2 and self.cumulative_movement > 2.0:
            self.up_movement_count += 1
            self.down_movement_count = 0
            return 1
        elif slow_down_moves >= 2 and self.cumulative_movement < -2.0:
            self.down_movement_count += 1
            self.up_movement_count = 0
            return -1
        else:
            # No clear sustained movement
            if abs(total_movement) < 3:
                self.up_movement_count = max(0, self.up_movement_count - 1)
                self.down_movement_count = max(0, self.down_movement_count - 1)
            return 0
    
    def _is_movement_confirmed(self, direction: int) -> bool:
        """Check if movement in given direction is confirmed by sustained frames"""
        fast_threshold = self.sustained_frames_fast
        slow_threshold = self.sustained_frames_slow
        
        if direction == 1:  # Up movement
            if abs(self.cumulative_movement) > self.movement_confirmation_threshold * 1.5:
                return self.up_movement_count >= fast_threshold
            else:
                return self.up_movement_count >= slow_threshold
                
        elif direction == -1:  # Down movement
            if abs(self.cumulative_movement) > self.movement_confirmation_threshold * 1.5:
                return self.down_movement_count >= fast_threshold
            else:
                return self.down_movement_count >= slow_threshold
        return False
    
    def _detect_state_change(self, current_angle: float) -> ExerciseState:
        """Detect state changes based on sustained movement and angle thresholds"""
        if len(self.angle_history) < 3:
            return self.state
        
        current_time = time.time()
        movement_direction = self._get_sustained_movement_direction()
          
        # Enforce minimum time in each phase
        # But allow faster transitions for very clear movements
        min_time_required = self.min_phase_duration
        if abs(self.cumulative_movement) > self.movement_confirmation_threshold * 2:
            min_time_required *= 0.6  # Reduce time requirement for very clear movements
        
        time_in_current_state = current_time - self.state_change_time
        if time_in_current_state < min_time_required:
            return self.state
        
        # State transition logic for shoulder abduction
        if self.state == ExerciseState.NEUTRAL:
            # Start UP phase with confirmed upward movement (arm lifting)
            if (movement_direction == 1 and 
                self._is_movement_confirmed(1) and 
                current_angle > self.min_angle_threshold + 15):
                return ExerciseState.UP_PHASE
                
        elif self.state == ExerciseState.UP_PHASE:
            # Go to DOWN phase with confirmed downward movement OR at peak
            if (movement_direction == -1 and self._is_movement_confirmed(-1)) or \
               (current_angle > self.max_angle_threshold):
                return ExerciseState.DOWN_PHASE
                
        elif self.state == ExerciseState.DOWN_PHASE:
            # Go to NEUTRAL when back at resting position
            if current_angle <= self.min_angle_threshold + 10:
                return ExerciseState.NEUTRAL
            # OR go to UP phase if strong upward movement confirmed
            elif movement_direction == 1 and self._is_movement_confirmed(1):
                return ExerciseState.UP_PHASE
        
        return self.state
    
    def update(self, angles_dict: Dict[str, Optional[float]]) -> Dict:
        """Update detector with new angle measurements"""
        raw_angle = angles_dict.get(self.angle_key)
        
        if raw_angle is None:
            return {
                'exercise': 'shoulder Abduction',
                'arm': self.arm.title(),
                'rep_count': self.rep_count,
                'state': 'No Detection',
                'angle': None,
                'feedback': f'{self.arm.title()} shoulder not visible',
                'debug_info': 'Camera cannot see shoulder'
            }
        
        current_angle = raw_angle  
        current_time = time.time()
        
        # Update angle history
        self._update_angle_history(current_angle)
        
        # Get movement direction
        movement_direction = self._get_sustained_movement_direction()
        
        # Detect state changes
        new_state = self._detect_state_change(current_angle)
        
        # Handle state transitions
        if new_state != self.state:
            self._handle_state_transition(new_state, current_angle, current_time)
        
        # Update angle tracking
        self._update_angle_tracking(current_angle)
        
        # Update last angle
        self.last_angle = current_angle
        
        return {
            'exercise': 'shoulder Abduction',
            'arm': self.arm.title(),
            'rep_count': self.rep_count,
            'state': self.state.value.replace('_', ' ').title(),
            'angle': round(current_angle, 1),
            'peak_angle': round(self.peak_angle, 1) if self.peak_angle else None,
            'valley_angle': round(self.valley_angle, 1) if self.valley_angle else None,
            'debug_info': {
                'movement_direction': movement_direction,
                'up_count': self.up_movement_count,
                'down_count': self.down_movement_count,
                'up_confirmed': self._is_movement_confirmed(1),
                'down_confirmed': self._is_movement_confirmed(-1),
                'cumulative_movement': round(self.cumulative_movement, 1),
                'recent_angles': [round(a, 1) for a in self.angle_history[-3:]] if len(self.angle_history) >= 3 else []
            }
        }
    
    def _handle_state_transition(self, new_state: ExerciseState, angle: float, current_time: float):
        """Handle transitions between exercise states"""
        
        if new_state == ExerciseState.UP_PHASE and self.state == ExerciseState.NEUTRAL:
            # Starting a new rep
            self.current_rep = RepData(start_time=current_time)
            self.peak_angle = angle
            self.valley_angle = angle
            print(f"DEBUG: Started new shoulder abduction rep at angle {angle:.1f} ({self.arm} arm)")
            
        elif new_state == ExerciseState.UP_PHASE and self.state == ExerciseState.DOWN_PHASE:
            # Going back up
            print(f"DEBUG: Continuing upward shoulder movement at {angle:.1f} ({self.arm} arm)")
            
        elif new_state == ExerciseState.DOWN_PHASE and self.state == ExerciseState.UP_PHASE:
            # Reached peak, starting descent
            if self.current_rep:
                self.current_rep.max_angle = self.peak_angle
            print(f"DEBUG: Shoulder peak reached at {self.peak_angle:.1f}, starting descent ({self.arm} arm)")
                
        elif new_state == ExerciseState.NEUTRAL and self.state == ExerciseState.DOWN_PHASE:
            # Completed rep
            if self.current_rep:
                # Check for minimum range of motion (shoulder abduction should have good range)
                range_of_motion = (self.peak_angle or 0) - (self.valley_angle or angle)
                if range_of_motion > 40:  # Minimum valid range for shoulder abduction
                    self.current_rep.complete_rep(current_time)
                    self.current_rep.min_angle = self.valley_angle or angle
                    self.rep_history.append(self.current_rep)
                    self.rep_count += 1
                    print(f"DEBUG: Shoulder abduction rep {self.rep_count} completed! Range: {self.valley_angle:.1f} to {self.peak_angle:.1f} ({self.arm} arm)")
                else:
                    print(f"DEBUG: Shoulder movement too small to count as rep (range: {range_of_motion:.1f}) ({self.arm} arm)")
                self.current_rep = None
                
            # Reset peak/valley for next rep
            self.peak_angle = None
            self.valley_angle = None
        
        # Update state and timing
        self.state = new_state
        self.state_change_time = current_time
        
        # Reset movement counters on state change
        self.up_movement_count = 0
        self.down_movement_count = 0
        self.cumulative_movement = 0
    
    def _update_angle_tracking(self, angle: float):
        """Update peak and valley angle tracking"""
        if self.state == ExerciseState.UP_PHASE:
            if self.peak_angle is None or angle > self.peak_angle:
                self.peak_angle = angle
                
        elif self.state == ExerciseState.DOWN_PHASE:
            if self.valley_angle is None or angle < self.valley_angle:
                self.valley_angle = angle
    
    def get_stats(self) -> Dict:
        """Get exercise statistics"""
        if not self.rep_history:
            return {
                'total_reps': 0,
                'avg_duration': 0,
                'avg_range_of_motion': 0,
                'arm': self.arm
            }
        
        durations = [rep.duration for rep in self.rep_history if rep.duration]
        ranges = [(rep.max_angle - rep.min_angle) for rep in self.rep_history 
                 if rep.max_angle is not None and rep.min_angle is not None]
        
        return {
            'total_reps': len(self.rep_history),
            'avg_duration': round(sum(durations) / len(durations), 1) if durations else 0,
            'avg_range_of_motion': round(sum(ranges) / len(ranges), 1) if ranges else 0,
            'last_rep_duration': round(self.rep_history[-1].duration, 1) if self.rep_history[-1].duration else 0,
            'arm': self.arm
        }
    
    def reset(self):
        """Reset the detector to initial state"""
        self.state = ExerciseState.NEUTRAL
        self.rep_count = 0
        self.current_rep = None
        self.rep_history = []
        self.last_angle = None
        self.peak_angle = None
        self.valley_angle = None
        self.angle_history = []
        self.up_movement_count = 0
        self.down_movement_count = 0
        self.current_movement_direction = 0
        self.cumulative_movement = 0
        print(f"Reset {self.arm} arm shoulder abduction detector")
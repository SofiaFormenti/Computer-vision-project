import math
import time

class LeftHandController:
    """
    Controls audio effects using left hand gestures:
    - Extended fingers (2-5) select which effect to control
    - Pinch gesture (thumb-index distance) controls the effect value
    """
    
    # Effect modes mapped to finger counts
    MODE_REVERB = 2      # 2 fingers = Reverb wet/dry (0-100%)
    MODE_FILTER = 3      # 3 fingers = Filter cutoff (200-8000 Hz)
    MODE_SPEED = 5       # 5 fingers = Playback speed (0.5x-2.0x)
    
    # OSC identifiers sent to Pure Data (must match PD patch)
    OSC_EFFECT_IDS = {
        MODE_REVERB: 4,  # Reverb is controlled by OSC ID 4
        MODE_FILTER: 3,  # Filter is controlled by OSC ID 3
        MODE_SPEED: 5,   # Speed is controlled by OSC ID 5
    }
    
    # Effect parameter ranges
    EFFECT_RANGES = {
        MODE_REVERB: (1.5, 0.05),# 5-100% wet
        MODE_FILTER: (10, 10000),# 200Hz - 8000Hz
        MODE_SPEED: (4.0, 1.0) # 1x- 4.0x speed
    }
    
    # Effect names for display
    EFFECT_NAMES = {
        MODE_REVERB: "REVERB",
        MODE_FILTER: "FILTER",
        MODE_SPEED: "SPEED"
    }
    
    def __init__(self, pd_sender, active_track=1):
        """
        Initialize the left hand controller.
        
        Args:
            pd_sender: PdSender instance to send OSC messages
            active_track: Which track to apply effects to (default: 1)
        """
        self.pd_sender = pd_sender
        self.active_track = active_track  # Which track is being controlled
        
        self.current_mode = None          # Current effect mode (2-5)
        self.last_pinch_value = None      # Last sent value to avoid spam
        self.last_send_time = 0           # Timestamp of last OSC send
        
        # Smoothing/stability
        self.pinch_history = []           # Store recent pinch distances
        self.HISTORY_SIZE = 5             # Number of samples to average
        self.MIN_SEND_INTERVAL = 0.05     # Min time between sends (50ms)
        
        # Pinch distance thresholds (in normalized coordinates)
        self.PINCH_MIN = 0.02             # Fingers touching
        self.PINCH_MAX = 0.15             # Fingers fully extended
        
    def set_active_track(self, track_number):
        """Change which track the effects are applied to."""
        self.active_track = track_number
        print(f"Left hand now controlling Track {track_number}")
    
    def reset(self):
        """Reset the controller to its initial state."""
        self.current_mode = None
        self.last_pinch_value = None
        self.last_send_time = 0
        self.pinch_history.clear()
        self.active_track = 1 # Or a default track
        print("Left hand controller has been reset.")


    def detect_pinch_distance(self, landmarks):
        """
        Calculate Euclidean distance between thumb tip and index finger tip.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Float distance in normalized coordinates (0.0-1.0 range)
        """
        # Landmark indices
        thumb_tip = landmarks.landmark[4]   # Thumb tip
        index_tip = landmarks.landmark[8]   # Index finger tip
        
        # Calculate Euclidean distance in 2D (x, y)
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + 
            (thumb_tip.y - index_tip.y) ** 2
        )
        
        return distance
    
    def smooth_pinch_distance(self, raw_distance):
        """
        Apply moving average filter to reduce jitter.
        
        Args:
            raw_distance: Current pinch distance measurement
            
        Returns:
            Smoothed distance value
        """
        # Add new value to history
        self.pinch_history.append(raw_distance)
        
        # Keep only recent history
        if len(self.pinch_history) > self.HISTORY_SIZE:
            self.pinch_history.pop(0)
        
        # Return average
        return sum(self.pinch_history) / len(self.pinch_history)
    
    def map_distance_to_value(self, distance, mode):
        """
        Map pinch distance to effect parameter value.
        
        Args:
            distance: Smoothed pinch distance (0.02-0.15 typical range)
            mode: Effect mode (2-5) to determine output range
            
        Returns:
            Mapped value appropriate for the effect type
        """
        # Normalize distance to 0-1 range
        normalized = (distance - self.PINCH_MIN) / (self.PINCH_MAX - self.PINCH_MIN)
        
        # Clamp to 0-1
        normalized = max(0.0, min(1.0, normalized))
        
        # Get the range for this effect
        min_val, max_val = self.EFFECT_RANGES[mode]
        
        # Map to effect range
        value = min_val + (normalized * (max_val - min_val))
        
        return value
    
    def update_mode_from_finger_count(self, finger_count):
        """
        Update the current effect mode based on extended fingers.
        
        Args:
            finger_count: Number of extended fingers on left hand
            
        Returns:
            True if mode changed, False otherwise
        """
        # Only update mode if finger count is 2, 3, or 5
        if finger_count in [2, 3, 5]:
            if self.current_mode != finger_count:
                self.current_mode = finger_count
                self.pinch_history.clear()  # Reset smoothing on mode change
                print(f"Left hand mode: {self.EFFECT_NAMES[finger_count]}")
                return True
        elif finger_count == 4:
            # 4 fingers is a neutral state, no effect
            self.current_mode = None
        
        return False
    
    def process_pinch(self, landmarks, finger_count):
        """
        Main processing function: detect pinch and send effect values.
        
        Args:
            landmarks: MediaPipe hand landmarks
            finger_count: Number of extended fingers
            
        Returns:
            Tuple of (mode, value) if sent, or (None, None) if not
        """
        # Update mode based on finger count
        self.update_mode_from_finger_count(finger_count)
        
        # If no mode selected yet, do nothing
        if self.current_mode is None:
            return None, None
        
        # Detect pinch distance
        raw_distance = self.detect_pinch_distance(landmarks)
        
        # Smooth the distance
        smooth_distance = self.smooth_pinch_distance(raw_distance)
        
        # Map to effect value
        effect_value = self.map_distance_to_value(smooth_distance, self.current_mode)

       #if self.current_mode == self.MODE_REVERB:
            #effect_value *= 2.0
        
        # Check if enough time has passed since last send
        current_time = time.time()
        if current_time - self.last_send_time < self.MIN_SEND_INTERVAL:
            return None, None
        
        # Check if value changed significantly (avoid sending duplicate values)
        if self.last_pinch_value is not None:
            # Different thresholds for different effects
            if self.current_mode == self.MODE_FILTER:
                threshold = 100  # 100 Hz change
            elif self.current_mode == self.MODE_REVERB:
                threshold = 0.02  # 2% change
            else:  # SPEED
                threshold = 0.05  # 0.05x change
            
            if abs(effect_value - self.last_pinch_value) < threshold:
                return None, None
        
        # Send the effect value to Pure Data
        self.send_effect(self.active_track, self.current_mode, effect_value)
        
        # Update tracking variables
        self.last_pinch_value = effect_value
        self.last_send_time = current_time
        
        return self.current_mode, effect_value
    
    def send_effect(self, track, effect_type, value):
        """
        Send effect parameter to Pure Data via OSC.
        
        Args:
            track: Track number to affect
            effect_type: Effect mode (2-5)
            value: Effect parameter value
        """
        # Get the correct OSC identifier for the current effect mode
        osc_id = self.OSC_EFFECT_IDS[effect_type]
        
        # Send OSC message: /effect [track] [effect_type] [value]
        self.pd_sender.send_effect(track, osc_id, value)
        
        # Print for debugging
        effect_name = self.EFFECT_NAMES[effect_type]
        print(f"→ Track {track} | {effect_name} (OSC ID: {osc_id}): {value:.2f}")
    
    def get_display_info(self):
        """
        Get formatted info for on-screen display.
        
        Returns:
            Dictionary with display information
        """
        if self.current_mode is None:
            return {
                'mode': 'NONE',
                'value': 0,
                'percentage': 0,
                'active': False
            }
        
        effect_name = self.EFFECT_NAMES[self.current_mode]
        value = self.last_pinch_value if self.last_pinch_value else 0
        
        # Calculate percentage for progress bar
        min_val, max_val = self.EFFECT_RANGES[self.current_mode]
        percentage = ((value - min_val) / (max_val - min_val)) * 100
        
        return {
            'mode': effect_name,
            'value': value,
            'percentage': percentage,
            'active': True,
            'track': self.active_track
        }
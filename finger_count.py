import cv2
import time
import mediapipe as mp

"""
The following code aims at leveraging the Gesture Recognizer of Mediapipe Solutions, an open source Google project
that operates on image data with a machine learning model, and accepts either static data or a continuous stream. 
The task outputs hand landmarks in image coordinates, hand landmarks in world coordinates, handedness (left/right hand), 
and the hand gesture categories of multiple hands.

In this case the output needed is a customized one, since the objective is to recognize the hand and the fingers of both
left and right hand. 

The class FingerCounter is a five-state state machine that uses MediaPipe to detect hand gestures and count fingers for 
a two-stage selection interface (instrument -> track)
"""



class FingerCounter:

    """
    Detects and processes both hands
    - RIGHT : track selection (state machine)
    - LEFT : effect control


    5 states for the right hand:
    0 -> waits for the closed hand to activate
    1 -> waits for user to select fingers to choose the instrument
    2 -> waits for closed fist to confirm instrument
    3 -> waits for user to select fingers to choose the track
    4 -> waits for closed fist to confirm track
    The arrival at the forth state makes the track start playing
    """

    WAITING_HAND_READY = 0
    WAITING_FOR_SETTING = 1
    WAITING_FOR_SETTING_CONFIRM = 2
    WAITING_FOR_OPTION = 3
    WAITING_FOR_OPTION_CONFIRM = 4

    def __init__(self, on_selection, on_restart, left_hand_controller = None):
        self.on_selection = on_selection
        self.left_hand_controller = left_hand_controller
        self.on_restart = on_restart

        self.state = self.WAITING_HAND_READY    # initialize state machine with 1st state
        self.selected_setting = None            # stores the instrument number
        self.selected_option = None             # stores the track number

        self.mp_hands = mp.solutions.hands          # Mediapipe for hand detection
        self.mp_draw = mp.solutions.drawing_utils   # Mediapipe for visual hand landmarks
        self.hands = self.mp_hands.Hands(           # create hand detector (2 hands + confidence level 70%)
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        """
        Stabilization filtering to prevent jitter readings
        The stabilization filter is key to making this usable
        -> prevents accidental selections from brief hand movements, without it the program is inconsistent
        """
        self.last_finger_count = None
        self.last_change_time = time.time()
        self.STABLE_TIME = 0.25   # time in sec for a reading to be considered valid

        # Restart button properties
        self.RESTART_BUTTON_RECT = (680, 10, 110, 40) # x, y, w, h

    def reset(self):
        """Resets the state machine for the right hand."""
        print("Resetting right hand state machine.")
        self.state = self.WAITING_HAND_READY
        self.selected_setting = None
        self.selected_option = None

    """
    Finger counting
    
    args: 
        - lm := Mediapipe hand landmarks
        - label := right or left to indicate which hand

    returns:
        integer count od extended fingers 
    """

    def count_fingers(self, lm, label):
        tips = [4, 8, 12, 16, 20]       # landmark indices for fingertips (order: thumb -> index -> middle -> ring -> pinky)
        fingers = []                    # stores boolean values (true/false) to see whether each finger is extended 

        
        """
        Thumb detection - logic is different if left or right hand
        -> right thumb is extended if the tip is to the left of the joint below it
        -> left thumb if tip to the right of joint below it
        """

        if label == "Right":
            fingers.append(lm.landmark[tips[0]].x < lm.landmark[tips[0] - 1].x)
        else:
            fingers.append(lm.landmark[tips[0]].x > lm.landmark[tips[0] - 1].x)

        # For the other fingers the finger is extended if the tip is above the joint 2 positions below it comparing y coords (lower y is higher on the screen)
        for i in range(1, 5):
            fingers.append(lm.landmark[tips[i]].y < lm.landmark[tips[i] - 2].y)

        return fingers.count(True)      # number of extended fingers

    def is_hand_closed(self, n):
        return n == 0


    """
    Stabilization filter -> filters the raw finger count to only return stable 
    readings preventing accidental triggers from brief finger movements.
        
    Args:
        raw_count: The current raw finger count from detection
            
    Returns:
        The stable finger count if held long enough, None otherwise
    """
    def get_stable_finger_count(self, raw_count):
        current_time = time.time()

        if raw_count != self.last_finger_count:
            self.last_change_time = current_time
            self.last_finger_count = raw_count

        # return stable value only if held long enough
        if current_time - self.last_change_time > self.STABLE_TIME:
            return self.last_finger_count

        return None  # if not stable yet return None, still stabilizing
    
    # -------------------------------------------------------
    # DISPLAY HELPERS
    # -------------------------------------------------------
    def draw_right_hand_ui(self, frame, stable_count):
        """Draw UI elements for right hand state machine."""
        # State-specific instructions
        if self.state == self.WAITING_HAND_READY:
            cv2.putText(frame, "RIGHT: Show CLOSED hand to activate",
                        (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        elif self.state == self.WAITING_FOR_SETTING:
            cv2.putText(frame, "RIGHT: Raise fingers for INSTRUMENT",
                        (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        elif self.state == self.WAITING_FOR_SETTING_CONFIRM:
            cv2.putText(frame, f"RIGHT: Close to confirm Instrument {self.selected_setting}",
                        (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)

        elif self.state == self.WAITING_FOR_OPTION:
            cv2.putText(frame, "RIGHT: Raise fingers for TRACK",
                        (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)

        elif self.state == self.WAITING_FOR_OPTION_CONFIRM:
            cv2.putText(frame, f"RIGHT: Close to confirm Track {self.selected_option}",
                        (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,150,0), 2)

        # Show stable finger count
        if stable_count is not None:
            cv2.putText(frame, f"RIGHT stable: {stable_count}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)

    def draw_left_hand_ui(self, frame, left_count):
        """Draw UI elements for left hand effect control."""
        if self.left_hand_controller is None:
            return
        
        # Get display info from left hand controller
        info = self.left_hand_controller.get_display_info()
        
        # Draw left hand finger count
        cv2.putText(frame, f"LEFT fingers: {left_count}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,100), 2)
        
        # Draw current effect mode and value
        if info['active']:
            mode_text = f"LEFT MODE: {info['mode']} (Track {info['track']})"
            cv2.putText(frame, mode_text,
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,255), 2)
            
            # Draw value
            if info['mode'] == 'FILTER':
                value_text = f"Value: {info['value']:.0f} Hz"
            elif info['mode'] in ['VOLUME', 'REVERB']:
                value_text = f"Value: {info['value']:.0%}"
            else:  # SPEED
                value_text = f"Value: {info['value']:.2f}x"
            
            cv2.putText(frame, value_text,
                        (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,255), 2)
            
            # Draw progress bar
            bar_x = 10
            bar_y = 210
            bar_width = 300
            bar_height = 20
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Filled portion
            fill_width = int((info['percentage'] / 100) * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + fill_width, bar_y + bar_height), 
                         (255, 100, 255), -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (255, 255, 255), 2)

    def draw_restart_button(self, frame):
        """Draws the restart button on the frame."""
        # Get frame dimensions to position button in the top-right
        h_frame, w_frame, _ = frame.shape
        
        # Update button rectangle based on frame width
        btn_w, btn_h = 110, 40
        btn_x = w_frame - btn_w - 10 # 10px padding from the right edge
        btn_y = 10                   # 10px padding from the top
        self.RESTART_BUTTON_RECT = (btn_x, btn_y, btn_w, btn_h)

        x, y, w, h = self.RESTART_BUTTON_RECT

        # Create a transparent overlay for the button
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 180), -1) # Dark red background
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame) # Blend with 50% opacity

        # Draw border and text on top of the blended rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2) # White border
        cv2.putText(frame, "RESTART", (x + 10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def mouse_callback(self, event, x, y, flags, param):
        """Handles mouse click events to check for restart button press."""
        if event == cv2.EVENT_LBUTTONDOWN:
            rx, ry, rw, rh = self.RESTART_BUTTON_RECT
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                print("Restart button clicked!")
                self.on_restart()


    # -------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------
    def run(self):
        """Main loop: capture video, process both hands."""
        cv2.namedWindow("Dual Hand Controller")
        cv2.setMouseCallback("Dual Hand Controller", self.mouse_callback)
        cap = cv2.VideoCapture(0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # Initialize hand data for this frame
            right_count_raw = None
            left_count_raw = None
            left_landmarks = None

            # -------------------------------------------
            # HAND DETECTION
            # -------------------------------------------
            if results.multi_hand_landmarks:
                for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handed.classification[0].label

                    # Process RIGHT hand (track selection)
                    if label == "Right":
                        right_count_raw = self.count_fingers(lm, "Right")

                    # Process LEFT hand (effect control)
                    elif label == "Left":
                        left_count_raw = self.count_fingers(lm, "Left")
                        left_landmarks = lm

                    # Draw landmarks for both hands
                    self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

            # -------------------------------------------
            # RIGHT HAND: STATE MACHINE LOGIC
            # -------------------------------------------
            stable_count = None
            if right_count_raw is not None:
                stable_count = self.get_stable_finger_count(right_count_raw)

            if stable_count is not None:
                # STATE 0: Wait for closed hand
                if self.state == self.WAITING_HAND_READY:
                    if self.is_hand_closed(stable_count):
                        print("Right hand ready.")
                        self.state = self.WAITING_FOR_SETTING

                # STATE 1: Select instrument
                elif self.state == self.WAITING_FOR_SETTING:
                    if stable_count > 0:
                        self.selected_setting = stable_count
                        print(f"Instrument selected: {self.selected_setting}")
                        self.state = self.WAITING_FOR_SETTING_CONFIRM

                # STATE 2: Confirm instrument
                elif self.state == self.WAITING_FOR_SETTING_CONFIRM:
                    if self.is_hand_closed(stable_count):
                        print(f"Instrument {self.selected_setting} confirmed.")
                        self.state = self.WAITING_FOR_OPTION

                # STATE 3: Select track
                elif self.state == self.WAITING_FOR_OPTION:
                    if stable_count > 0:
                        self.selected_option = stable_count
                        print(f"Track selected: {self.selected_option}")
                        self.state = self.WAITING_FOR_OPTION_CONFIRM

                # STATE 4: Confirm track
                elif self.state == self.WAITING_FOR_OPTION_CONFIRM:
                    if self.is_hand_closed(stable_count):
                        print(f"Track {self.selected_option} confirmed!")
                        
                        # Callback for track selection
                        self.on_selection(self.selected_setting, self.selected_option)
                        
                        # Update left hand controller to control this track
                        if self.left_hand_controller:
                            self.left_hand_controller.set_active_track(self.selected_option)
                        
                        # Reset state machine
                        self.state = self.WAITING_HAND_READY
                        self.selected_setting = None
                        self.selected_option = None

            # -------------------------------------------
            # LEFT HAND: EFFECT CONTROL
            # -------------------------------------------
            if self.left_hand_controller and left_landmarks and left_count_raw:
                # Process pinch gesture for effect control
                self.left_hand_controller.process_pinch(left_landmarks, left_count_raw)

            # -------------------------------------------
            # DRAW UI
            # -------------------------------------------
            # Right hand UI
            if right_count_raw is not None:
                cv2.putText(frame, f"RIGHT raw: {right_count_raw}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            self.draw_right_hand_ui(frame, stable_count)
            
            # Left hand UI
            if left_count_raw is not None:
                self.draw_left_hand_ui(frame, left_count_raw)
            
            # Restart Button
            self.draw_restart_button(frame)

            # Show frame
            cv2.imshow("Dual Hand Controller", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

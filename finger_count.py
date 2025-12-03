import cv2
import time
import mediapipe as mp

"""
The following code aims at leveraging the Gesture Recognizer of Mediapipe Solutions, an open source Google project
that operates on image data with a machine learning (ML) model, and accepts either static data or a continuous stream. 
The task outputs hand landmarks in image coordinates, hand landmarks in world coordinates, handedness (left/right hand), 
and the hand gesture categories of multiple hands.

In this case the output needed is a customized one, since the objective is to recognize the hand and the fingers of both
left and right hand. 

The class FingerCounter is a five-state state machine that uses MediaPipe to detect hand gestures and count fingers for 
a two-stage selection interface (instrument -> track)
"""



class FingerCounter:

    """
    5 states:
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

    def __init__(self, on_selection):
        self.on_selection = on_selection

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


    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)                          # flip for mirror effect
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # convert from BGR (OpenCV default) to RGB (MediaPipe requirement)
            results = self.hands.process(rgb)                   # process with Mediapipe to detect hands

            right_count_raw = None                              # initialize variable to store right hand finger count for this frame

            
            # check if any hands where detected, loop through them, get the label indicating whether its L or R
            if results.multi_hand_landmarks:
                for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handed.classification[0].label

                    if label == "Right":
                        right_count_raw = self.count_fingers(lm, "Right")

                    self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

            # apply stabilization filter
            stable_count = None
            if right_count_raw is not None:
                stable_count = self.get_stable_finger_count(right_count_raw)


            # state machine logic
            if stable_count is not None:
                # -----------------------------
                # STATE 0 — WAIT FOR CLOSED HAND (ACTIVATE)
                # -----------------------------
                if self.state == self.WAITING_HAND_READY:
                    cv2.putText(frame, "Show CLOSED right hand to activate",
                                (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

                    if self.is_hand_closed(stable_count):
                        print("Right hand ready.")
                        self.state = self.WAITING_FOR_SETTING

                # -----------------------------
                # STATE 1 — SELECT SETTING
                # -----------------------------
                elif self.state == self.WAITING_FOR_SETTING:
                    cv2.putText(frame, "Raise fingers to choose INSTRUMENT",
                                (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    if stable_count > 0:
                        self.selected_setting = stable_count
                        print(f"Instrument selected: {self.selected_setting}")
                        self.state = self.WAITING_FOR_SETTING_CONFIRM

                # -----------------------------
                # STATE 2 — CONFIRM SETTING
                # -----------------------------
                elif self.state == self.WAITING_FOR_SETTING_CONFIRM:
                    cv2.putText(frame, "Close hand to CONFIRM instrument",
                                (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,50), 2)

                    if self.is_hand_closed(stable_count):
                        print(f"Instrument {self.selected_setting} confirmed.")
                        self.state = self.WAITING_FOR_OPTION

                # -----------------------------
                # STATE 3 — SELECT OPTION
                # -----------------------------
                elif self.state == self.WAITING_FOR_OPTION:
                    cv2.putText(frame, "Raise fingers to choose TRACK",
                                (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,0), 2)

                    if stable_count > 0:
                        self.selected_option = stable_count
                        print(f"Track selected: {self.selected_option}")
                        self.state = self.WAITING_FOR_OPTION_CONFIRM

                # -----------------------------
                # STATE 4 — CONFIRM OPTION
                # -----------------------------
                elif self.state == self.WAITING_FOR_OPTION_CONFIRM:
                    cv2.putText(frame, "Close hand to CONFIRM track",
                                (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,150,0), 2)

                    if self.is_hand_closed(stable_count):
                        print(f"Track {self.selected_option} confirmed!")

                        # call callback function with both selections
                        self.on_selection(self.selected_setting, self.selected_option)

                        # reset state machine to initial state
                        self.state = self.WAITING_HAND_READY
                        self.selected_setting = None
                        self.selected_option = None

            # Display unfiltered finger count on screen if detected
            if right_count_raw is not None:
                cv2.putText(frame, f"Right hand (raw): {right_count_raw}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Display the filtered finger count on screen (if available)
            if stable_count is not None:
                cv2.putText(frame, f"Stable fingers: {stable_count}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)

            cv2.imshow("Hand Selector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

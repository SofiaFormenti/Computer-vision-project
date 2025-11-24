import cv2
import mediapipe as mp

class FingerCounter:

    WAITING_FOR_SETTING = 0
    WAITING_FOR_OPTION  = 1

    def __init__(self, on_selection):
        self.on_selection = on_selection
        self.state = self.WAITING_FOR_SETTING
        self.selected_setting = None
        self.selected_option = None

        self.stable_frames = 0
        self.last_finger_count = None
        self.confirmed_finger_count = None
        self.REQUIRED_STABLE_FRAMES = 10   # adjust for more/less stability

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # -----------------------------
    # FINGER COUNT LOGIC
    # -----------------------------

    
    def count_fingers(self, lm, label):
        tips = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if label == "Right":
            fingers.append(lm.landmark[tips[0]].x < lm.landmark[tips[0] - 1].x)
        else:
            fingers.append(lm.landmark[tips[0]].x > lm.landmark[tips[0] - 1].x)

        # Other 4 fingers
        for i in range(1, 5):
            fingers.append(lm.landmark[tips[i]].y < lm.landmark[tips[i] - 2].y)

        return fingers.count(True)

    def hand_closed(self, finger_count):
        return finger_count == 0
    
    

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            right_count = "-"

            if results.multi_hand_landmarks:
                for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handed.classification[0].label

                    finger_count = self.count_fingers(lm, label)
                    if label == "Right":
                        right_count = finger_count

            # ----------------------------------------
            # STATE MACHINE
            # ----------------------------------------
            if right_count != "-":

                if self.state == self.WAITING_FOR_SETTING:
                    if right_count > 0:
                        self.selected_setting = right_count

                    if self.hand_closed(right_count) and self.selected_setting:
                        print(f"[Confirmed] Instrument {self.selected_setting}")
                        self.state = self.WAITING_FOR_OPTION

                elif self.state == self.WAITING_FOR_OPTION:
                    if right_count > 0:
                        self.selected_option = right_count

                    if self.hand_closed(right_count) and self.selected_option:
                        print(f"[Confirmed] Track {self.selected_option} for Instrument {self.selected_setting}")

                        # callback to main
                        self.on_selection(self.selected_setting, self.selected_option)

                        # reset and go back to setting selection
                        self.state = self.WAITING_FOR_SETTING
                        self.selected_setting = None
                        self.selected_option = None

            cv2.putText(frame, f"Right: {right_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("Hand Selector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

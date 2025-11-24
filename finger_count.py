# finger_count.py
import cv2
import time
import mediapipe as mp

class FingerCounter:

    # States
    WAITING_HAND_READY = 0
    WAITING_FOR_SETTING = 1
    WAITING_FOR_SETTING_CONFIRM = 2
    WAITING_FOR_OPTION = 3
    WAITING_FOR_OPTION_CONFIRM = 4

    def __init__(self, on_selection):
        self.on_selection = on_selection

        self.state = self.WAITING_HAND_READY
        self.selected_setting = None
        self.selected_option = None

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Stability filtering
        self.last_finger_count = None
        self.last_change_time = time.time()
        self.STABLE_TIME = 0.25   # how long the finger must remain stable

    # -------------------------------------------------------
    # FINGER COUNTING
    # -------------------------------------------------------
    def count_fingers(self, lm, label):
        tips = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if label == "Right":
            fingers.append(lm.landmark[tips[0]].x < lm.landmark[tips[0] - 1].x)
        else:
            fingers.append(lm.landmark[tips[0]].x > lm.landmark[tips[0] - 1].x)

        # Other fingers
        for i in range(1, 5):
            fingers.append(lm.landmark[tips[i]].y < lm.landmark[tips[i] - 2].y)

        return fingers.count(True)

    def is_hand_closed(self, n):
        return n == 0

    # -------------------------------------------------------
    # STABILIZATION FILTER
    # -------------------------------------------------------
    def get_stable_finger_count(self, raw_count):
        current_time = time.time()

        if raw_count != self.last_finger_count:
            self.last_change_time = current_time
            self.last_finger_count = raw_count

        # return stable value only if held long enough
        if current_time - self.last_change_time > self.STABLE_TIME:
            return self.last_finger_count

        return None  # still stabilizing

    # -------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            right_count_raw = None

            # -------------------------------------------
            # HAND DETECTION
            # -------------------------------------------
            if results.multi_hand_landmarks:
                for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handed.classification[0].label

                    if label == "Right":
                        right_count_raw = self.count_fingers(lm, "Right")

                    self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

            # -------------------------------------------
            # APPLY STABILIZATION FILTER
            # -------------------------------------------
            stable_count = None
            if right_count_raw is not None:
                stable_count = self.get_stable_finger_count(right_count_raw)

            # -------------------------------------------
            # STATE MACHINE LOGIC
            # -------------------------------------------
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

                        # final callback
                        self.on_selection(self.selected_setting, self.selected_option)

                        # reset
                        self.state = self.WAITING_HAND_READY
                        self.selected_setting = None
                        self.selected_option = None

            # Display finger count
            if right_count_raw is not None:
                cv2.putText(frame, f"Right hand (raw): {right_count_raw}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if stable_count is not None:
                cv2.putText(frame, f"Stable fingers: {stable_count}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)

            cv2.imshow("Hand Selector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

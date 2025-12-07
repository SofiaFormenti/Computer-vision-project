import math
import time

class SimpleLeftHandController:
    """
    Very simple gesture system:
    - 1 finger  => select FILTER
    - 2 fingers => select REVERB
    - Close hand (0 fingers) = OK / confirm / next step
    - Pinch controls the effect value during ADJUSTING mode
    """

    EFFECT_FILTER = 1
    EFFECT_REVERB = 2

    def __init__(self, pd_sender,active_track=1):
        self.pd = pd_sender

        self.state = "IDLE"           # IDLE → SELECTED → ADJUSTING
        self.selected_effect = None   # FILTER or REVERB
        self.last_value = 0.0

        # pinch thresholds
        self.PINCH_MIN = 0.02
        self.PINCH_MAX = 0.15

    def pinch_distance(self, lm):
        thumb = lm.landmark[4]
        index = lm.landmark[8]
        return math.dist((thumb.x, thumb.y), (index.x, index.y))

    def normalize(self, d):
        n = (d - self.PINCH_MIN) / (self.PINCH_MAX - self.PINCH_MIN)
        return max(0.0, min(1.0, n))

    def update(self, landmarks, finger_count):
        """
        Call this every frame with left-hand landmarks and finger count.
        """

        # -----------------------
        # STATE: IDLE
        # -----------------------
        if self.state == "IDLE":
            if finger_count == 1:
                self.selected_effect = self.EFFECT_FILTER
                print("Selected: FILTER (show OK gesture to confirm)")
                self.state = "SELECTED"

            elif finger_count == 2:
                self.selected_effect = self.EFFECT_REVERB
                print("Selected: REVERB (show OK gesture to confirm)")
                self.state = "SELECTED"

            return

        # -----------------------
        # STATE: SELECTED (waiting for OK gesture)
        # -----------------------
        if self.state == "SELECTED":
            if finger_count == 0:   # closed hand = OK
                print("Confirmed. Start pinching to adjust value.")
                self.state = "ADJUSTING"
            return

        # -----------------------
        # STATE: ADJUSTING (pinch controls the effect)
        # -----------------------
        if self.state == "ADJUSTING":

            if finger_count == 0:
                print("Finished adjustment. Returning to IDLE.")
                self.state = "IDLE"
                return

            # If pinch is available, adjust
            d = self.pinch_distance(landmarks)
            value = self.normalize(d)

            if abs(value - self.last_value) > 0.01:
                self.last_value = value

                if self.selected_effect == self.EFFECT_FILTER:
                    self.pd.send_effect(3, value)  # cutoff control
                    print(f"FILTER cutoff = {value:.2f}")

                elif self.selected_effect == self.EFFECT_REVERB:
                    self.pd.send_effect(4, value)  # reverb control
                    print(f"REVERB amount = {value:.2f}")

            return

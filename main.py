"""
Main script for dual-hand gesture control:
- RIGHT hand: Select instrument and track
- LEFT hand: Control effects on selected track
"""

from finger_count import FingerCounter
from left_hand_controller import LeftHandController
from pd_sender import PdSender

# -----------------------------
# INITIALIZE
# -----------------------------

# Create Pure Data OSC sender
pd = PdSender()

# Create left hand controller for effect control
left_controller = LeftHandController(pd_sender=pd, active_track=1)

# -----------------------------
# TRACK SELECTION CALLBACK (Right Hand)
# -----------------------------
def on_selection(setting, option):
    """
    Called when user confirms instrument + track selection with right hand.
    
    Args:
        setting: Instrument number (1-5)
        option: Track number (1-5)
    """
    print(f"\n{'='*50}")
    print(f"TRACK SELECTED: Instrument {setting}, Track {option}")
    print(f"{'='*50}\n")
    
    # Send selection to Pure Data
    pd.send_selection(setting, option)
    
    # Update left hand controller to control this track
    left_controller.set_active_track(option)

# -----------------------------
# RUN THE APPLICATION
# -----------------------------
if __name__ == "__main__":
    print("Starting Dual Hand Gesture Controller...")
    print("\nRIGHT HAND: Track Selection")
    print("  1. Close hand to activate")
    print("  2. Raise fingers (1-5) to select instrument")
    print("  3. Close hand to confirm")
    print("  4. Raise fingers (1-5) to select track")
    print("  5. Close hand to confirm")
    print("\nLEFT HAND: Effect Control")
    print("  - 2 fingers: Volume control")
    print("  - 3 fingers: Filter cutoff control")
    print("  - 4 fingers: Reverb mix control")
    print("  - 5 fingers: Playback speed control")
    print("  - Pinch thumb+index: Adjust effect value")
    print("\nPress 'q' to quit\n")
    
    # Create and run finger counter with both hands
    fc = FingerCounter(
        on_selection=on_selection,
        left_hand_controller=left_controller
    )
    fc.run()
"""
Main script for dual-hand gesture control:
- RIGHT hand: Select instrument and track
- LEFT hand: Control effects on selected track
"""

from finger_count import FingerCounter
from left_hand_controller import LeftHandController
#from simplified_left_hand_controller import SimpleLeftHandController
from pd_sender import PdSender

# -----------------------------
# PD VERSION 2
# -----------------------------
from audio_player import AudioPlayer
import audio_player
print(">>> MAIN IMPORTED AUDIO_PLAYER FROM:", audio_player.__file__)


pd=PdSender()
audio= AudioPlayer()

left_controller = LeftHandController(pd_sender=pd, active_track=1)
#left_controller = SimpleLeftHandController(pd_sender=pd, active_track=1)

# --- NEW: Keep track of which tracks are currently playing ---
active_tracks = set() # e.g., {(1, 2), (3, 4)}

def on_selection(setting, option):
    print(f"\n{'='*50}")
    print(f"TRACK SELECTED: Instrument {setting}, Track {option}")
    print(f"{'='*50}\n")

    key = (setting, option)

    # --- NEW: Toggle the track's playback state ---
    if key in active_tracks:
        # If track is already playing, stop it and remove it from the set
        print(f"Toggling OFF: Instrument {setting}, Track {option}")
        audio.stop_loop(setting, option)
        active_tracks.remove(key)
    else:
        # If track is not playing, start it and add it to the set
        print(f"Toggling ON: Instrument {setting}, Track {option}")
        audio.play_loop(setting, option)
        active_tracks.add(key)

    # Send selection to Pure Data (optional)
    pd.send_selection(setting, option)
    # Update effect controller for this track
    left_controller.set_active_track(option)


def restart_simulation():
    """Stops all audio and resets all controllers to their initial state."""
    print("\n" + "="*20 + " RESTARTING SIMULATION " + "="*20 + "\n")
    # Stop all playing audio tracks
    audio.stop_all_loops()
    active_tracks.clear()

    # Reset the left hand controller
    left_controller.reset()
    
    # Reset the right hand controller (state machine)
    # The FingerCounter instance 'fc' is not available here, so we need it to reset itself.
    # We will call a reset method on it from within its class on restart trigger.
    fc.reset()

# -----------------------------
# PD WHOLE VERSION 1






# -----------------------------
# PD WHOLE VERSION 1
# -----------------------------
# Create Pure Data OSC sender
#pd = PdSender()

# Create left hand controller for effect control
#left_controller = LeftHandController(pd_sender=pd, active_track=1)

# -----------------------------
# TRACK SELECTION CALLBACK (Right Hand)
# -----------------------------
#def on_selection(setting, option):
    """
    Called when user confirms instrument + track selection with right hand.
    
    Args:
        setting: Instrument number (1-5)
        option: Track number (1-5)
    """
    #print(f"\n{'='*50}")
    #print(f"TRACK SELECTED: Instrument {setting}, Track {option}")
    #print(f"{'='*50}\n")
    
    # Send selection to Pure Data
    #pd.send_selection(setting, option)
    
    # Update left hand controller to control this track
    #left_controller.set_active_track(option)

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
    print("Click the 'RESTART' button in the top-right to reset everything.")
    
    # Create and run finger counter with both hands
    fc = FingerCounter(
        on_selection=on_selection,
        on_restart=restart_simulation,
        left_hand_controller=left_controller
    )
    fc.run()

import os
print("Répertoire de travail :", os.getcwd())
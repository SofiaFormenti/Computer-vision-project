"""
Main script for dual-hand gesture control:
- RIGHT hand: Select instrument and track
- LEFT hand: Control effects on selected track
"""

from finger_count import FingerCounter
from left_hand_controller import LeftHandController
from pd_sender import PdSender
from audio_player import AudioPlayer
import audio_player

print(">>> MAIN IMPORTED AUDIO_PLAYER FROM:", audio_player.__file__)


pd=PdSender()
audio= AudioPlayer()

left_controller = LeftHandController(pd_sender=pd, active_track=1)


active_tracks = set()   # tracks that are currently playing, for example {(1, 2), (3, 4)}

def on_selection(setting, option):
    print(f"\n{'='*50}")
    print(f"TRACK SELECTED: Instrument {setting}, Track {option}")
    print(f"{'='*50}\n")

    key = (setting, option)

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

    pd.send_selection(setting, option)
    left_controller.set_active_track(option)


def restart_simulation():
    """Stops all audio and resets all controllers to their initial state."""
    print("\n" + "="*20 + " RESTARTING SIMULATION " + "="*20 + "\n")

    audio.stop_all_loops()
    active_tracks.clear()

    left_controller.reset()
    
    fc.reset()


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
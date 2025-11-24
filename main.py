from finger_count import FingerCounter
from audio_player import AudioPlayer

# -----------------------------
# MAP SETTINGS & OPTIONS → LOOP FILES
# -----------------------------
LOOP_FILES = {
    (1, 1): "loops/jazz-drums-loop.wav",
    # Example:
    # (1, 2): "loops/piano-loop.wav",
    # (2, 1): "loops/synth-loop.wav",
    # Add more here...
}

player = AudioPlayer()

def on_selection(setting, option):
    """Called automatically when user confirms a setting+option."""
    key = (setting, option)

    if key in LOOP_FILES:
        player.play_loop(setting, option, LOOP_FILES[key])
    else:
        print(f"No loop assigned to Instrument {setting}, Track {option}")

# Run the finger detector
fc = FingerCounter(on_selection)
fc.run()

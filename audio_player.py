from pydub import AudioSegment
import simpleaudio as sa
import threading

class AudioPlayer:
    def __init__(self):
        self.active_loops = {}   # {(setting, option): play_obj}

    def play_loop(self, setting, option, filepath):
        """Play an audio loop forever in background."""
        sound = AudioSegment.from_file(filepath)
        sound_data = sound.raw_data
        channels = sound.channels
        sample_width = sound.sample_width
        frame_rate = sound.frame_rate

        def loop_forever():
            while True:
                play_obj = sa.play_buffer(sound_data, channels, sample_width, frame_rate)
                play_obj.wait_done()

        thread = threading.Thread(target=loop_forever, daemon=True)
        thread.start()

        self.active_loops[(setting, option)] = thread
        print(f"▶️  Started loop for Instrument {setting}, Track {option}")

import os
import simpleaudio as sa
import threading
from pydub import AudioSegment


class AudioPlayer:
    def __init__(self):

        self.BASE = os.path.dirname(os.path.abspath(__file__))

        def p(*paths):
            return os.path.join(self.BASE, *paths)

        # Registry: instrument -> 4 tracks
        self.tracks = {
            1: [
                p("samples", "fixed_instr1_piano1.wav"),
                p("samples", "fixed_instr1_piano2.wav"),
                p("samples", "fixed_instr1_piano3.wav"),
                p("samples", "fixed_instr1_piano4.wav"),
            ],
            2: [
                p("samples", "fixed_instr2_drum1.wav"),
                p("samples", "fixed_instr2_drum2.wav"),
                p("samples", "fixed_instr2_drum3.wav"),
                p("samples", "fixed_instr2_drum4.wav"),
            ],
            3: [
                p("samples", "fixed_instr3_bass1.wav"),
                p("samples", "fixed_instr3_bass2.wav"),
                p("samples", "fixed_instr3_bass3.wav"),
                p("samples", "fixed_instr3_bass4.wav"),
            ],
            4: [
                p("samples", "fixed_instr4_guit1.wav"),
                p("samples", "fixed_instr4_guit2.wav"),
                p("samples", "fixed_instr4_guit3.wav"),
                p("samples", "fixed_instr4_guit4.wav"),
            ],
        }

        self.active_loops = {}   # key = (instr, track)
        self.stop_flags = {}     # key = (instr, track)


    def stop_loop(self, instrument, track):
        key = (instrument, track)
        if key in self.stop_flags:
            print(f"Stopping loop {key}")
            self.stop_flags[key] = True

    def stop_all_loops(self):
        """Stop all currently playing loops."""
        print("Stopping all audio loops...")
        # Create a copy of the keys to iterate over, as we'll be modifying the dict
        active_loop_keys = list(self.active_loops.keys())
        for key in active_loop_keys:
            self.stop_loop(key[0], key[1])
        
        # Wait for all threads to finish
        for key in active_loop_keys:
            if key in self.active_loops and self.active_loops[key].is_alive():
                self.active_loops[key].join() # Wait for the thread to terminate

        self.active_loops.clear()


    def play_loop(self, instrument, track):
        key = (instrument, track)

        # stop the previous loop of the same slot
        self.stop_loop(instrument, track)

        fp = self.tracks[instrument][track - 1]
        print(f"▶ Starting loop {key}: {fp}")
        sound = AudioSegment.from_wav(fp)

        raw = (
            sound.raw_data,
            sound.channels,
            sound.sample_width,
            sound.frame_rate
        )

        self.stop_flags[key] = False

        def loop():
            while not self.stop_flags[key]:
                play_obj = sa.play_buffer(*raw)
                play_obj.wait_done()

        th = threading.Thread(target=loop, daemon=True)
        th.start()

        self.active_loops[key] = th
        print(f"✔ Loop active: {key}")

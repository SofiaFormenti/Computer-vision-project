<div align="center">
  
# Real Time Interactive DJ set 

CSC_51073_EP class of Computer vision 2025 

This project aims to use hand recognition to make real time dj-set by converting to MIDI.


</div>

## Features

| Gesture | Action |
|---------|--------|
| ✋ Left Hand | 
| 🤏 Pinch & Drag | Change Amplitude of effect |
| Right Hand ✋|
| ✊ Closed Fist | Confirm Selection |
| ✌️ Finger indication | Select Instrument and Tracks |


## Architecture

```
Computer-vision-project/
├── Hand recognition.py             # opencv implementation
├── audio_player.py
├── finger_count.py
├── left_hand_controller.py
├──main.py
├──pd_sender.py
├──prog_test.pd
├──requirements.txt
├──test.pd
├── __pycache__/
│   ├── finger_count.cpython-312.pyc 
│   ├── left_hand_controller.cpython-312.py
│   └── pd_sender.cpython-312.pyc
├── samples/
│   ├── funkydrum.wav
│   ├── instr1_bass1.wav
│   └── instr1_bass2.wav
│   └──instr1_bass3.wav
│   └──instr1_bass4.wav
│   └──instr2_drum1.wav
│   └──instr2_drum2.wav
│   └──instr2_drum3.wav
│   └──instr2_drum4.wav
│   └──instr3_piano1.wav
│   └──instr3_piano2.wav
│   └──instr3_piano3.wav
│   └──jazz-drums-loop.wav

```

<div align="center">
  
# Real Time Interactive DJ set 

CSC_51073_EP class of Computer vision 2025 

This project aims to use hand recognition to make real time dj-set by converting to MIDI.

 ‼️Work in progress‼️

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
├── Hand recognition.py             # opencv implementation, run this file for the convex hull implementation
├── audio_player.py
├── finger_count.py                 # Right hand instrument and track selection
├── left_hand_controller.py         # Left hand Effects controller with pinch
├── main.py                         # run this for the Mediapipe implementation
├── pd_sender.py
├── prog_test.pd
├── requirements.txt
├── test.pd
├── __pycache__/
│   ├── finger_count.cpython-312.pyc 
│   ├── left_hand_controller.cpython-312.py
│   └── pd_sender.cpython-312.pyc
└── samples/
   ├── funkydrum.wav
   ├── instr1_bass1.wav
   └── instr1_bass2.wav
   └──instr1_bass3.wav
   └──instr1_bass4.wav
   └──instr2_drum1.wav
   └──instr2_drum2.wav
   └──instr2_drum3.wav
   └──instr2_drum4.wav
   └──instr3_piano1.wav
   └──instr3_piano2.wav
   └──instr3_piano3.wav
   └──jazz-drums-loop.wav

```


## Track Overview


## 🎹 Instrument 1 — Track 1

**Mood:** Dreamlike  
**BPM:** 78  
**Tags:** ambient, soft, floaty  
**Description:**  
A gentle atmospheric pad designed to create a dreamy, floating texture.

---

## 🥁 Instrument 2 — Track 1

**Mood:** Energetic  
**BPM:** 95  
**Tags:** jazz, drums, rhythm  
**Description:**  
Jazz-style drum loop with light swing, suitable for layering under ambient pads.


## Instruments

### 🎹 Instrument 1 — Pads
| Track | Mood       | Notes               |
|-------|------------|---------------------|
| 1     | Dreamlike  | Soft evolving pads  |
| 2     | Ambient    | Long reverb tail    |

### 🥁 Instrument 2 — Drums
| Track | Mood       | Notes               |
|-------|------------|---------------------|
| 1     | Energetic  | Jazz drum loop      |

### 🎸 Instrument 3 — Bass
| Track | Mood       | Notes               |
|-------|------------|---------------------|
| 1     | Dark       | Synth bass line     |


## Key Features

- **Interactivity**: 

<div align="center">
  
# Real Time Interactive DJ set 

CSC_51073_EP class of Computer vision 2025 

By: Aline Baumberger and Sofia Formenti

This project aims to use hand recognition to make real time dj-set effects by using samples.

All the tracks you can choose from were made from scratch by Aline and Nathan! 💿

</div>

## Features

| Hand | Gesture | 
|---------|--------|
|  Left Hand |  Effects control (lowpass, reverb, speed)
| 🤏 Pinch & Drag | Change Amplitude of effect |
| Right Hand |  Instrument and Track choice
| ✌️ Finger indication | Select Instrument and Track number |
| ✊ Closed Fist | Confirm Selection |

## How to use

Install the required libraries, [Pure Data](https://puredata.info/downloads) and [VB-Cable](https://vb-audio.com/Cable/). 
<br />
In your computer's sound settings select as audio output the virtual cable.
Open Pure Data, go into Media and then into Audio Settings and select as input device the VB-Cable.
Now tick the box next to DSP, you should see "Active audio". Open the file simpler_version_pd.pd run main.py and have fun with our virtual dj-set!
<br />
<br />
![PureData](/img/PD_settings.jpeg)

To run the finger count implementation done completely with OpenCV without Mediapipe, simply run Hand recognition.py
<br />
Beware: this is not connected to pure data, so sadly you will hear no tracks!😔

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
   ├──instr1_piano1.wav
   ├──instr1_piano2.wav
   └──instr1_piano3.wav
   └──instr1_piano4.wav
   └──instr2_drum1.wav
   └──instr2_drum2.wav
   └──instr2_drum3.wav
   └──instr2_drum4.wav
   └──instr3_bass1.wav
   └──instr3_bass2.wav
```

## Track Overview


## Instrument 1 - Piano 

**Track 1:** Ambiant chords  
**BPM:** 120  
**Tags:** ambient, soft, floaty  

**Track 2:** Arpeggio  
**BPM:** 120  
**Tags:** - 

**Track 3:** Ambiant chords  
**BPM:** 120  
**Tags:** -

**Track 4:** Ambiant chords  
**BPM:** 120  
**Tags:** -

---

##  Instrument 2 - Drums

**Track 1:** Energetic  
**BPM:** 95  
**Tags:** drums, rhythm  
**Description:**  
Jazz-style drum loop with light swing, suitable for layering under ambient pads.

**Track 2:** Energetic  
**BPM:** 95  
**Tags:** drums, rhythm  
**Description:**  


**Track 3:** Energetic  
**BPM:** 95  
**Tags:** drums, rhythm  
**Description:**  


**Track 4:** Energetic  
**BPM:** 95  
**Tags:** drums, rhythm  
**Description:**  


---

##  Instrument 3 - bass

**Track 1:** Energetic  
**BPM:** 95  
**Tags:** drums, rhythm  
**Description:** 


**Track 2:** Energetic  
**BPM:** 95  
**Tags:** drums, rhythm  
**Description:**  




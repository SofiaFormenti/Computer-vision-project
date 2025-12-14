<div align="center">
  
# Real Time Interactive DJ set 

CSC_51073_EP class of Computer vision 2025 

By: Aline Baumberger and Sofia Formenti

This project aims to use hand recognition to make real time dj-set effects by using samples.

All the tracks you can choose from were made from scratch by Aline and Nathan! 💿

</div>

## Demo

![Demo]([Demo.mp4]([https://youtu.be/hevnI0u5mno](https://www.youtube.com/watch?v=hevnI0u5mno)))

## Features

| Hand | Gesture | 
|---------|--------|
|  Left Hand |  Effects control (lowpass, reverb, speed)
| 🤏 Pinch & Drag | Change Amplitude of effect |
| Right Hand |  Instrument and Track choice
| ✌️ Finger indication | Select Instrument and Track number |
| ✊ Closed Fist | Confirm Selection |

## How to use

Install the required libraries found in requirements.txt, [Pure Data](https://puredata.info/downloads) and [VB-Cable](https://vb-audio.com/Cable/). 
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
├── Hand recognition.py             # opencv implementation, run this file for the convexity defect-based implementation
├── audio_player.py                 # oversees track play control
├── finger_count.py                 # right hand instrument and track selection
├── left_hand_controller.py         # left hand effects controller with pinch
├── main.py                         # run this for the Mediapipe implementation
├── pd_sender.py
├── prog_test.pd
├── requirements.txt
├── test.pd
└── samples/
   ├──conv_instr1_piano1.wav
   ├──conv_instr1_piano2.wav
   └──conv_instr1_piano3.wav
   └──conv_instr1_piano4.wav
   └──conv_instr2_drum1.wav
   └──conv_instr2_drum2.wav
   └──conv_instr2_drum3.wav
   └──conv_instr2_drum4.wav
   └──conv_instr3_bass1.wav
   └──conv_instr3_bass2.wav
   └──conv_instr4_guit1.wav
   └──conv_instr4_guit2.wav
   └──conv_instr4_guit3.wav
```

Audio files are here converted to WAV PCM 16-bit format to ensure a standard, uncompressed, and widely supported representation.
This guarantees consistent encoding and sampling rates across all files, preventing compatibility issues during audio processing in Pure Data.

If new audio files are added from digital audio workstations (such as Ableton Live), it is recommended to convert them beforehand so that all audio data remains consistent !




## Track Overview

All tracks share the same key (E minor) and the same tempo (120 BPM) to ensure rythmic and tonal coherence.
However, not all tracks fit equally well together, as the goal was to allow combinations across 4 piano tracks, 4 drum tracks, 3 guitar tracks, and 2 bass tracks (13 tracks total), which makes achieving perfect compatibility non-trivial.

For this reason, priority is given to drum, guitar, and piano tracks that blend particularly well together, all of which were created by Aline and Nathan using Ableton Live.

| Instrument | Track | Style / Role        | Description                  |
|------------|-------|---------------------|-----------------------|
| Piano      | 1     | Ambient chords      | Sustained atmophseric and ambiant chords ambient |
| Piano      | 2     | Arpeggio            |     Simple  distorted arpeggio chords progression                    |
| Piano      | 3     | Distorted chords       |          Sparse chord progression with reverb/ texture             |
| Piano      | 4     | Rythmic arpeggio chords      |   Fast and stacatto arpeggio with dynamic layering                     |
| Drums      | 1     | Rythmic           | Tight rythmic pattern with kick, hihat and snare      |
| Drums      | 2     | Slower, groovier           | Jazz-inspired drum lopp with swing, light hihat and reverbed snares        |
| Drums      | 3     | Slowed Drum n Bass           | Percussive loop with breaks and syncopation, slowed jungle type DnB         |
| Drums      | 4     | Energetic           | Percussive pattern, more afro style with percs and rims         |
| Bass       | 1     | Dark, electronic arpeggio           |   Electronic Acid type techno pattern       |
| Bass       | 2     | Distorted simple bass line       | Distorted sparse  descending bass linegroove          |
| Guitar     | 1     |    Bright electric guitare arpeggio                 | Clean electric guitar arpeggios with modulation effects               |
| Guitar     | 2     |      Slower guitar chords               | Electric guitar chords with passing tones                |
| Guitar     | 3     |   Distorted guitar                  | Descending electric guitar pattern, rock like              |
"""

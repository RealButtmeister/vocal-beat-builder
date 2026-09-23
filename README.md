# Vocal Beat Builder

Extract vocals locally, generate changing editable drum MIDI patterns at a chosen BPM, and align vocal attacks to the actual drum hits. Version 1.2.1 includes the corrected first-section lead-in timing.

## Run from source (Windows)

Install Python 3.13 with Tkinter and the Windows Python launcher, then run in this folder:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe beat_app.py
```

After setup, `Start.bat` uses the local `.venv`. See [USER_GUIDE.md](USER_GUIDE.md) for controls and workflow.

Install the external audio tools described in [tools/INSTALL.md](tools/INSTALL.md). This is a small source checkout; no model weights, audio files or Python runtime are included.

First full-song use downloads its local separation/classification dependencies and models. Allow internet access for setup and several GB of free space. Processing runs locally on the CPU. See [PANNs notices.txt](PANNs%20notices.txt) for the adapted classifier and label attribution.

The combined drum MIDI uses FL Studio C5–E5 (MIDI 60–64): kick, snare, off-snare, closed hat, open hat. Four base patterns vary between sections. Instrument placeholders contain a short starter note, not recovered instrument melodies. Instrument names and section boundaries are estimates.

## Source and distribution

This repository contains editable application source. Generated exports, private settings, API keys, user audio, bundled runtimes and packaged executables are excluded. Original local releases remain separate. No new license is granted for original application code in this publication; dependency notices retain their own terms.

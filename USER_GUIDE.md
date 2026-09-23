# User guide

These notes describe the application. For this source checkout, use README.md and Start.bat instead of the original packaged launchers or executable.

```text
VOCAL BEAT BUILDER 1.2.1
Song -> vocal extraction -> changing drum patterns -> vocal aligned to drum hits

START
Extract the whole folder. Double-click Start Vocal Beat Builder.vbs.
This is a separate app. Your existing Vocal Grid Snap and FLP Setup Builder
remain unchanged, including the backed-up builder version.

1. Browse for a downloaded Songer song (WAV, MP3, FLAC, M4A or similar).
2. Leave Input type at Full song, or choose Already isolated vocal for a stem.
3. Enter the target BPM for FL Studio. Source BPM can stay blank for automatic
   estimation from the separated drums; enter the known original BPM if needed.
4. Half-time gives a dubstep-style backbeat. Straight uses beats 2 and 4.
5. Click Go. Then Play preview to hear the aligned vocal with simple drum sounds.
6. Open exports and follow FL Studio - Start Here.txt.

FIRST FULL-SONG RUN
The app downloads and installs local separation and instrument-recognition
engines and models on first use.
An internet connection and several GB of free disk space are needed for setup.
The private .runtime folder does not replace your other Python installations.
Models are cached in this app's models folder. Later runs reuse them.
Audio stays on this computer; no Songer login or audio upload is needed.
This first version uses the CPU. Separating a complete song can take several
minutes or longer. The window stays responsive, and Cancel stops its worker.
If setup is interrupted, press Go again to retry. Keep the entire app folder.

FIVE DRUM LANES
C5  / MIDI 60 = Kick
C#5 / MIDI 61 = Snare
D5  / MIDI 62 = Off-snare (quieter ghost/secondary snare)
D#5 / MIDI 63 = Closed hat
E5  / MIDI 64 = Open hat

Drums_Combined.mid puts all five lanes in one note track, with tempo/section
markers in a conductor track. This is YOUR note mapping, not General MIDI drums.
Map your own drum instrument to these notes. Drums_By_Lane.mid is also included
if you prefer five separate named tracks.

There are four distinct two-bar patterns: Main A, Main B, Side A and Side B.
They change at section boundaries and recur through longer songs. Full songs
of at least eight source bars get at least four sections; very short excerpts
can use fewer. All four reusable patterns are always exported in Patterns.
Dense vocal sections can get extra quieter sixteenth hats. Exact section MIDI
is exported too, so these extra hits are present in the editable beat.

HOW VOCAL ALIGNMENT WORKS
The app first converts the vocal's overall tempo without changing its pitch.
It then finds syllable-like attacks and moves them to distinct, actual drum-hit
positions in each section. Strong drum hits get a modest preference. This is
not merely a uniform BPM grid: target positions come from the exported beat.
Original consonant attacks are retained; slice bodies are compressed only when
needed to avoid the following slice. Tiny fades prevent cut-edge clicks.

Sections are estimated from sound changes in the full song on a 4/4 bar grid,
with regular phrase boundaries as a fallback. They are not guaranteed verse or
chorus recognition. Section length is a maximum/adaptive phrase guide. Every
section reserves a whole even number of bars. First FL bar defaults to 2, so
section starts stay on even one-based FL bar numbers. A short ending is padded.
The short lead-in before the detected first beat is kept without adding an
extra two-bar section. This fixes an unintended pause near the first transition.

This does not transcribe lyrics. Some soft syllables can be missed, sustained
notes can be split, and stem separation can leave artifacts or background bleed.
Check the preview. Adjust sensitivity/minimum slice or override Source BPM if
the rhythm is wrong. Tempo estimates can be half or double the intended tempo.
Editing MIDI hit timing later in FL does not move the already-rendered vocal.

INSTRUMENT PLACEHOLDERS
A trained audio classifier checks short passages of the original song and its
separated accompaniment for instrument families. It no longer assigns names
from whole-song stem loudness, which could miss brief instrumental passages.
Names are estimates: quiet, layered or unusual sounds can still be missed or
misidentified. Unclear accompaniment is labelled Unidentified accompaniment.
The app cannot identify the exact synth, plugin or preset. Instruments.json
records the evidence, timestamps and model scores for checking the result.

Instrument_Placeholders.mid is one bar long and contains one short C5 note per
named instrument track, with nonzero velocity so FL Studio can import it.
It contains only those named tracks. Choose Create one
channel per track in FL's MIDI import window. These are starter notes only:
delete or replace them after choosing your sounds. They are not transcribed
melodies. Instrument_MIDI contains separate one-bar files for dragging onto
individual instruments. Each includes the same short C5 starter note.
When FL is installed, Instrument_Placeholders.flp is included with blank named
Sampler channels, so you have actual placeholders to replace with your synths.
Opening this FLP starts a separate project; save your current work first.
Instruments.txt always provides the same list. Vocal-input mode also checks for
any accompaniment still audible in the supplied audio. If no instrument can be
named confidently, the app supplies a clearly labelled generic placeholder.

For an existing export, click Recheck instruments... and choose its folder.
This updates the names and placeholder files without rerendering the vocal or
changing the drum MIDI. Previous instrument files are backed up inside that
export. Keep the original source audio available at its saved location.

EXPORTS
Vocal_ALIGNED.wav          The vocal to import at Playlist bar 1.
Vocal_EXTRACTED.wav        The extracted original vocal before realignment.
Vocal_WITH_DRUMS.wav       Listening preview with synthesized drum sounds.
Drums_PREVIEW.wav          The same simple preview drums without the vocal.
Drums_Combined.mid         Complete editable drum arrangement, notes C5-E5.
Drums_By_Lane.mid          Alternative drum export with five named note tracks.
Patterns/                 Four base patterns and exact per-section MIDI.
Instrument_Placeholders.*  Named starter-note MIDI and optional blank-channel FLP.
Instrument_MIDI/           Separate starter-note MIDI for each instrument.
Instruments.txt / .json    Names plus detailed recognition evidence.
Stems/                    Extracted source stems, for full-song input.
Sections.csv              Source ranges, pattern choices and FL bar positions.
Timing.csv                Each vocal attack and its destination drum MIDI tick.
Drum_Hits.csv              Drum notes, velocities, sections and exact timings.
Report.json               Settings, analysis estimates and notes to review.

All timeline exports start at the same project origin. Keep the vocal's leading
silence and put the full drum MIDI at bar 1 too. Set FL to the chosen BPM.
Every run gets a new folder. Your source file is never overwritten.

REQUIREMENTS / ERRORS
Windows and the existing Python 3.13 installation with NumPy, SciPy and SoundFile.
Install them using requirements.txt. FL Studio is needed
only for the optional placeholder FLP and your production work.
Install FFmpeg and Rubber Band as described in tools/INSTALL.md. Separation dependencies/models download locally.
First-run setup errors are recorded in Setup log.txt and Classifier setup log.txt.
Recognition details are recorded in Classifier log.txt. App errors are recorded
in Last error.txt. Try the .bat launcher if the window does not open.
```

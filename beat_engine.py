"""Local vocal-to-drum arrangement. Every vocal target is an exported MIDI hit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import math
import os
import re
import secrets
import shutil
import struct
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy import ndimage, signal
import vocal_engine as vocal

ROOT = Path(__file__).resolve().parent
LANES = ('Kick', 'Snare', 'Off-snare', 'Closed hat', 'Open hat')
NOTES = (60, 61, 62, 63, 64)  # FL Studio calls MIDI 60 C5.
PPQ = 960
PATTERN_NAMES = ('Main A', 'Main B', 'Side A', 'Side B')


def beat_frames(beats, sr, bpm):
    """One rounding rule for MIDI hits, audio rendering and section boundaries."""
    return np.rint(np.asarray(beats, dtype=np.float64) * (60. / float(bpm) * int(sr))).astype(np.int64)


@dataclass
class BeatConfig:
    target_bpm: float = 140.0
    source_bpm: float | None = None
    sensitivity: float = .6
    min_slice_ms: float = 80.0
    section_bars: int = 8
    start_bar: int = 2
    style: str = 'Half-time'
    input_is_vocal: bool = False
    seed: int = 0

    def validate(self):
        for name, low, high in [('target_bpm', 40, 240), ('sensitivity', 0, 1), ('min_slice_ms', 40, 500)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or not low <= float(value) <= high:
                raise ValueError(f'{name.replace("_", " ")} must be between {low} and {high}.')
        if self.source_bpm is not None and (not math.isfinite(float(self.source_bpm)) or not 30 <= self.source_bpm <= 300):
            raise ValueError('Source BPM must be blank or between 30 and 300.')
        if self.section_bars not in (4, 8, 16):
            raise ValueError('Choose 4, 8 or 16 bars for section length.')
        if not isinstance(self.start_bar, int) or not 1 <= self.start_bar <= 32:
            raise ValueError('First bar must be a whole number from 1 to 32.')
        if self.style not in ('Half-time', 'Straight'):
            raise ValueError('Choose Half-time or Straight drums.')
        if not isinstance(self.seed, int):
            raise ValueError('Pattern seed must be a whole number.')


def _mono(audio):
    return audio[:, int(np.argmax(np.mean(audio.astype(np.float64) ** 2, axis=0)))]


def _analysis(audio, sr):
    mono = _mono(audio)
    divisor = math.gcd(sr, 11025)
    mono = signal.resample_poly(mono, 11025 // divisor, sr // divisor)
    hop = 256
    if len(mono) < 1024:
        mono = np.pad(mono, (0, 1024 - len(mono)))
    frequencies, times, spectrum = signal.stft(mono, fs=11025, nperseg=1024,
                                               noverlap=1024-hop, boundary='zeros')
    magnitude = np.abs(spectrum)
    logmag = np.log1p(magnitude * 500)
    flux = np.maximum(np.diff(logmag, axis=1, prepend=logmag[:, :1]), 0).mean(axis=0)
    flux = np.maximum(flux - ndimage.median_filter(flux, size=21) * .7, 0)
    edges = np.geomspace(60, 5500, 13)
    features = np.asarray([logmag[(frequencies >= lo) & (frequencies < hi)].mean(axis=0)
                           for lo, hi in zip(edges, edges[1:])])
    return times, flux, features


def detect_tempo(audio, sr, preferred=140.):
    """Autocorrelation with an explicit tempo-octave preference, not a certainty."""
    times, flux, _ = _analysis(audio, sr)
    if len(times) < 10 or float(flux.max(initial=0)) < 1e-8:
        return preferred, 0., 0.
    step = float(times[1] - times[0])
    corr = signal.fftconvolve(flux, flux[::-1], mode='full')[len(flux)-1:]
    corr /= np.maximum(1, np.arange(len(flux), 0, -1))
    candidates = np.arange(60., 200.01, .25)
    lags = 60 / candidates / step
    scores = np.interp(lags, np.arange(len(corr)), corr)
    scores += .35 * np.interp(lags * 2, np.arange(len(corr)), corr)
    scores *= .7 + .3 * np.exp(-np.log2(candidates / preferred) ** 2 / .5)
    winner = int(np.argmax(scores))
    bpm = float(candidates[winner])
    # Beat trackers cannot uniquely distinguish half-time from double-time.
    # Use the user's requested tempo to choose the nearest equivalent octave,
    # avoiding an accidental 2x vocal speed-up for a 66/132 BPM song at 140.
    aliases = [bpm * factor for factor in (.5, 1., 2., 4.) if 30 <= bpm*factor <= 300]
    bpm = min(aliases, key=lambda value: abs(math.log2(value/preferred)))
    beat = 60 / bpm
    phases = np.linspace(0, beat, 80, endpoint=False)
    phase_scores = [float(np.interp(np.arange(phase, times[-1], beat), times, flux).sum()) for phase in phases]
    offset = float(phases[int(np.argmax(phase_scores))])
    confidence = float(np.clip((scores[winner] / max(float(np.median(scores)), 1e-9) - 1) / 3, 0, 1))
    return bpm, offset, confidence


def detect_sections(audio, sr, bpm, preferred_bars=8, beat_offset=0.):
    """Bar-level timbre changes, bounded fallback phrases; labels stay generic."""
    times, _, features = _analysis(audio, sr)
    bar_seconds = 240 / bpm
    duration = len(audio) / sr
    total_bars = max(1, math.ceil(max(0, duration-beat_offset) / bar_seconds))
    # A full song should actually use all four patterns. For shorter songs,
    # shorten the phrase cap so four two-bar-or-longer sections can fit.
    if total_bars >= 8:
        preferred_bars = min(preferred_bars, max(2, (total_bars // 8) * 2))
    bars = []
    for index in range(total_bars):
        left = 0 if index == 0 else beat_offset + index * bar_seconds
        right = beat_offset + (index + 1) * bar_seconds
        selection = (times >= left) & (times < right)
        bars.append(features[:, selection].mean(axis=1) if selection.any() else np.zeros(features.shape[0]))
    bars = np.asarray(bars)
    norms = np.linalg.norm(bars, axis=1, keepdims=True)
    normalized = bars / np.maximum(norms, 1e-8)
    novelty = np.zeros(total_bars)
    for index in range(2, total_bars-1):
        novelty[index] = np.linalg.norm(normalized[max(0,index-2):index].mean(axis=0)
                                        - normalized[index:min(total_bars,index+2)].mean(axis=0))
    threshold = float(np.median(novelty) + .65 * np.std(novelty))
    boundaries = [0]
    reasons = ['Start']
    while total_bars - boundaries[-1] > preferred_bars:
        previous = boundaries[-1]
        candidates = list(range(previous + min(4, preferred_bars), min(total_bars-2, previous+preferred_bars) + 1, 2))
        if not candidates:
            break
        best = max(candidates, key=lambda index: novelty[index])
        if novelty[best] > max(.08, threshold):
            boundaries.append(best)
            reasons.append('Estimated timbre change')
        else:
            boundaries.append(previous + preferred_bars)
            reasons.append('Regular phrase fallback')
    boundaries.append(total_bars)
    sections = []
    for i, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        start = 0. if left == 0 else min(duration, beat_offset + left * bar_seconds)
        end = duration if right == total_bars else min(duration, beat_offset + right * bar_seconds)
        if end <= start:
            continue
        sections.append({'section': i+1, 'source_start_seconds': start, 'source_end_seconds': end,
                         'source_grid_start_seconds': beat_offset + left * bar_seconds,
                         'source_grid_end_seconds': beat_offset + right * bar_seconds,
                         'source_bars': right-left, 'boundary_method': reasons[i],
                         'novelty': round(float(novelty[left]), 5),
                         'energy': float(norms[left:right].mean())})
    return sections


def make_patterns(style='Half-time', seed=0):
    """Four deliberately different two-bar patterns, always using the five lanes."""
    rng = np.random.default_rng(seed)
    patterns = {}
    kicks = [[0, 1.5, 3.25, 4, 5.75, 7], [0, .75, 1.75, 4, 4.75, 6.75],
             [0, 3, 4, 5.5], [0, 1.25, 3.5, 4, 5.25, 7.5]]
    hats = [[.5, 1, 1.75, 2.5, 3, 3.5, 4.5, 5, 5.5, 6.5, 7.25],
            [.25, .5, 1.25, 2.25, 2.75, 3.25, 4.25, 5.25, 5.5, 6.25, 6.5, 7],
            [.5, 1.25, 1.75, 2.5, 3.5, 4.75, 5.25, 6.5, 7.25],
            [.25, .5, .75, 1.5, 1.75, 2.25, 2.5, 3, 3.25, 4.5, 5, 5.5, 6.25, 6.5, 7, 7.25, 7.75]]
    ghosts = [[1.25, 5.5], [3.5, 7.5], [1.5, 5.75], [5.75, 6.75, 7.25]]
    opens = [[3.75, 7.75], [1.5, 5.75], [2.75, 6.75], [3.75, 7.5]]
    snares = [2, 6] if style == 'Half-time' else [1, 3, 5, 7]
    for index, name in enumerate(PATTERN_NAMES):
        hits = []
        for lane, positions in enumerate((kicks[index], snares, ghosts[index], hats[index], opens[index])):
            for beat in positions:
                velocity = [112, 116, 55, 72, 83][lane] + int(rng.integers(-6, 7))
                hits.append({'beat': float(beat), 'lane': lane, 'note': NOTES[lane],
                             'velocity': int(np.clip(velocity, 1, 127))})
        patterns[name] = sorted(hits, key=lambda hit: (hit['beat'], hit['lane']))
    return patterns


def ordered_hits(desired, candidates, salience=None):
    """Minimum-cost, strictly ordered assignment to distinct real drum hit times."""
    desired = np.asarray(desired, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if not len(desired):
        return np.array([], dtype=np.int64)
    if len(candidates) < len(desired) or np.any(np.diff(candidates) <= 0):
        raise ValueError('Not enough distinct drum hits for the vocal attacks.')
    n, m = len(desired), len(candidates)
    salience = np.ones(m) if salience is None else np.asarray(salience)
    cost = (desired[:, None] - candidates[None, :]) ** 2 + .003 * (1-salience)[None, :]
    previous = cost[0].copy()
    parents = np.full((n, m), -1, dtype=np.int32)
    for row in range(1, n):
        current = np.full(m, np.inf)
        best, best_index = np.inf, -1
        for column in range(row, m):
            prior = column - 1
            if previous[prior] < best:
                best, best_index = previous[prior], prior
            current[column] = best + cost[row, column]
            parents[row, column] = best_index
        previous = current
    choice = int(np.argmin(previous))
    selected = [choice]
    for row in range(n-1, 0, -1):
        choice = int(parents[row, choice])
        selected.append(choice)
    return np.asarray(selected[::-1], dtype=np.int64)


def _hit_candidates(hits):
    combined = {}
    for hit in hits:
        salience = (1., 1., .45, .15, .35)[hit['lane']]
        combined[hit['beat']] = max(salience, combined.get(hit['beat'], 0))
    positions = np.array(sorted(combined))
    return positions, np.asarray([combined[beat] for beat in positions])


def plan_arrangement(sections, anchors, sr, source_bpm, config, patterns):
    ratio = source_bpm / config.target_bpm
    beat_seconds = 60 / config.target_bpm
    next_bar = config.start_bar
    targets = np.full(len(anchors), -1, dtype=np.int64)
    hits, planned = [], []
    # Main A/B recur; the side patterns appear at transitions and contrast sections.
    cycle = ('Side A', 'Main A', 'Main B', 'Side B', 'Main A', 'Main B', 'Side A', 'Side B')
    if len(sections) < 4:
        cycle = ('Main A', 'Main B', 'Side A', 'Side B')
    for index, section in enumerate(sections):
        left = round(section['source_start_seconds'] * ratio * sr)
        right = round(section['source_end_seconds'] * ratio * sr)
        selection = np.flatnonzero((anchors >= left) & (anchors < right))
        # The first source range includes the short lead-in before the detected
        # downbeat. That pickup is not another two-bar phrase. Reserve the
        # musical bar count, keeping its beat-grid origin separate from the
        # full audio range used to retain every source sample.
        source_bars = section.get('source_bars', (right-left) / sr / beat_seconds / 4)
        bars = max(2, int(math.ceil(max(0, source_bars-1e-7)/2)*2))
        grid_left = round(section.get('source_grid_start_seconds', section['source_start_seconds']) * ratio * sr)
        name = cycle[index % len(cycle)]
        start_beat = (next_bar-1)*4
        local = []
        for bar in range(0, bars, 2):
            local.extend({**hit, 'beat': start_beat + bar*4 + hit['beat'], 'section': index+1}
                         for hit in patterns[name])
        desired = start_beat * beat_seconds + (anchors[selection]-grid_left) / sr
        positions, weights = _hit_candidates(local)
        added = 0
        needs_support = len(desired) > len(positions)
        if len(desired) and not needs_support:
            trial = ordered_hits(desired, positions*beat_seconds, weights)
            needs_support = float(np.max(np.abs(positions[trial]*beat_seconds-desired))) > .32
        if needs_support:
            # Dense delivery gets a quieter sixteenth-hat variation. These hits
            # are exported too: no target ever exists only in the quantizer.
            existing = set(positions.tolist())
            for step in range(bars*16):
                beat = start_beat + step*.25
                if beat not in existing:
                    local.append({'beat': beat, 'lane': 3, 'note': 63, 'velocity': 48,
                                  'section': index+1, 'support': True})
                    added += 1
            positions, weights = _hit_candidates(local)
        while len(desired) > len(positions):
            # Preserve all attacks even at extreme tempo changes. Report the
            # extra space; never pile multiple syllables onto one hit.
            for hit in patterns[name]:
                local.append({**hit, 'beat': start_beat + bars*4 + hit['beat'], 'section': index+1})
            bars += 2
            positions, weights = _hit_candidates(local)
        choices = ordered_hits(desired, positions*beat_seconds, weights)
        targets[selection] = beat_frames(positions[choices], sr, config.target_bpm)
        row = {**section, 'pattern': name, 'start_bar': next_bar, 'bars': bars,
               'end_bar_exclusive': next_bar+bars, 'target_start_seconds': start_beat*beat_seconds,
               'target_end_seconds': (start_beat+bars*4)*beat_seconds,
               'attacks': len(selection), 'support_hat_hits': added,
               'max_local_shift_ms': float(np.max(np.abs(positions[choices]*beat_seconds-desired), initial=0)*1000)}
        planned.append(row)
        hits.extend(local)
        next_bar += bars
    if len(targets) and (np.any(np.diff(targets) <= 0) or np.any(targets < 0)):
        raise ValueError('Vocal targets are not in strictly increasing order.')
    hits.sort(key=lambda hit: (hit['beat'], hit['lane']))
    return planned, hits, targets, (next_bar-1)*4


def _vlq(value):
    value = int(value)
    result = [value & 127]
    while value >> 7:
        value >>= 7
        result.insert(0, (value & 127) | 128)
    return bytes(result)


def _meta(kind, text):
    data = text.encode('utf-8')
    return bytes([255, kind]) + _vlq(len(data)) + data


def _track(events, end_tick):
    payload = bytearray()
    previous = 0
    for tick, priority, message in sorted(events, key=lambda entry: (entry[0], entry[1])):
        tick = max(previous, int(tick))
        payload.extend(_vlq(tick-previous))
        payload.extend(message)
        previous = tick
    payload.extend(_vlq(max(0, int(end_tick)-previous)) + b'\xff\x2f\x00')
    return b'MTrk' + struct.pack('>I', len(payload)) + payload


def write_midi(path, hits, bpm, total_beats, sections=(), split=False, placeholders=()):
    melodic_channels = [channel for channel in range(16) if channel != 9]
    if len(placeholders) > len(melodic_channels):
        raise ValueError('A placeholder MIDI can contain up to 15 instrument tracks.')
    conductor = [(0, 0, _meta(3, 'Vocal Beat Builder')),
                 (0, 1, b'\xff\x51\x03' + round(60_000_000/bpm).to_bytes(3, 'big')),
                 (0, 1, b'\xff\x58\x04\x04\x02\x18\x08')]
    for section in sections:
        tick = round((section['start_bar']-1)*4*PPQ)
        conductor.append((tick, 2, _meta(6, f"Section {section['section']:02}: {section['pattern']}")))
    # Instrument files contain only real named tracks. Put their tempo into
    # the first instrument instead of showing a fake "Vocal Beat Builder" lane.
    instrument_only = bool(placeholders) and not hits
    tracks = [] if instrument_only else [conductor]
    if hits:
        for lane in range(5) if split else [None]:
            track = [(0, 0, _meta(3, LANES[lane] if lane is not None else 'Combined drums C5-E5'))]
            for hit in hits:
                if lane is not None and hit['lane'] != lane:
                    continue
                tick = round(hit['beat']*PPQ)
                length = round((.2 if hit['lane'] == 4 else .1)*PPQ)
                track.extend([(tick, 4, bytes([0x99, hit['note'], hit['velocity']])),
                              (tick+length, 3, bytes([0x89, hit['note'], 0]))])
            tracks.append(track)
    for index, (name, channel) in enumerate(zip(placeholders, melodic_channels)):
        track = [(0, 0, _meta(3, name)),
                       (0, 1, _meta(1, 'Starter C5 note for import only. Delete or replace it; this is not a transcribed melody.')),
                       (0, 4, bytes([0x90 | channel, 60, 64])),
                       (PPQ, 3, bytes([0x80 | channel, 60, 0]))]
        if instrument_only and index == 0:
            track.extend(conductor[1:])
        tracks.append(track)
    data = b'MThd' + struct.pack('>IHHH', 6, 1, len(tracks), PPQ)
    data += b''.join(_track(track, round(total_beats*PPQ)) for track in tracks)
    Path(path).write_bytes(data)


def write_placeholder_midis(folder, names, bpm, total_beats, sections=(), combined_name='Instrument_Placeholders.mid'):
    """Named starter tracks plus individually draggable one-bar MIDI files."""
    folder = Path(folder)
    # These are channel/import helpers, not the full arrangement. A one-bar
    # file keeps its visible starter note from shrinking into a 4-minute view.
    write_midi(folder/combined_name, [], bpm, 4, placeholders=names)
    individual = folder/'Instrument_MIDI'
    individual.mkdir(exist_ok=True)
    for index, name in enumerate(names, 1):
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')[:80] or 'Instrument'
        write_midi(individual/f'{index:02d}_{safe}.mid', [], bpm, 4, placeholders=[name])
    return folder/combined_name


def render_drums(hits, frames, sr, bpm, event=None):
    rng = np.random.default_rng(1492)
    sounds = []
    for lane in range(5):
        duration = [.35, .19, .105, .055, .23][lane]
        t = np.arange(round(duration*sr)) / sr
        noise = rng.standard_normal(len(t))
        if lane == 0:
            wave = np.sin(2*np.pi*(47*t + (135-47)*.028*(1-np.exp(-t/.028)))) * np.exp(-t*15)
            wave += noise*.08*np.exp(-t*180)
        elif lane in (1, 2):
            filt = signal.sosfilt(signal.butter(2, 1000, 'highpass', fs=sr, output='sos'), noise)
            wave = (.65*filt + .28*np.sin(2*np.pi*185*t))*np.exp(-t*(27 if lane == 1 else 48))
        else:
            filt = signal.sosfilt(signal.butter(3, min(7000, sr*.35), 'highpass', fs=sr, output='sos'), noise)
            wave = filt * np.exp(-t*(85 if lane == 3 else 21)) * .35
        wave = wave.astype(np.float32)
        wave *= min(.7/max(float(np.max(np.abs(wave))), 1e-8), 1.)
        sounds.append(wave)
    output = np.zeros(frames, dtype=np.float32)
    closed = [int(beat_frames(hit['beat'], sr, bpm)) for hit in hits if hit['lane'] == 3]
    for index, hit in enumerate(hits):
        if index % 100 == 0:
            vocal.check_cancel(event)
        start = int(beat_frames(hit['beat'], sr, bpm))
        sound = sounds[hit['lane']]
        count = min(len(sound), frames-start)
        if hit['lane'] == 4:
            following = next((frame for frame in closed if frame > start), frames)
            count = min(count, following-start)
        if count > 0:
            output[start:start+count] += sound[:count] * hit['velocity']/127
    output *= min(.8/max(float(np.max(np.abs(output))), 1e-8), 1.)
    return np.repeat(output[:, None], 2, axis=1)


def identify_families(stems, mix_rms, mix_audio=None, mix_sr=None):
    """Find localized accompaniment activity; names here are separator labels."""
    measurements = []
    mix_windows = None
    if mix_audio is not None and mix_sr:
        mix_windows = np.array([np.sqrt(np.mean(mix_audio[i:i+mix_sr].astype(np.float64)**2))
                                for i in range(0, len(mix_audio), mix_sr)])
    for name, path in stems.items():
        if name in ('vocals', 'drums'):
            continue
        total, count, windows = 0., 0, []
        with sf.SoundFile(path) as handle:
            for block in handle.blocks(blocksize=handle.samplerate, dtype='float32', always_2d=True):
                total += float(np.sum(block.astype(np.float64)**2))
                count += block.size
                windows.append(float(np.sqrt(np.mean(block.astype(np.float64)**2))))
        levels = np.array(windows)
        reference = np.full(len(levels), mix_rms)
        if mix_windows is not None and len(mix_windows):
            reference[:min(len(reference),len(mix_windows))] = mix_windows[:len(reference)]
        active = np.flatnonzero((levels >= max(2e-5, mix_rms*.006)) & (levels >= reference*.08))
        if not len(active):
            continue
        peak = int(np.argmax(levels))
        measurements.append({'family': {'other': 'Unidentified accompaniment', 'piano': 'Piano or keys'}.get(name, name.title()),
                             'stem': name, 'rms': math.sqrt(total/max(count, 1)),
                             'active_seconds': int(len(active)), 'peak_seconds': peak,
                             'peak_rms': float(levels[peak]),
                             'evidence_seconds': active.tolist(),
                             'status': 'Localized stem activity; exact instrument name requires classification'})
    return measurements


def detect_instrument_names(source, stems, mix, sr, progress, cancel_event):
    from instrument_classifier import classify_instruments
    mix_rms = float(np.sqrt(np.mean(mix.astype(np.float64)**2)))
    activity = identify_families(stems, mix_rms, mix, sr)
    accompaniment = [stems[name] for name in ('bass', 'guitar', 'piano', 'other') if name in stems]
    result = classify_instruments(source, progress=progress, cancel_event=cancel_event,
                                  accompaniment_paths=accompaniment or None)
    families = list(result.get('families', []))
    # A separated "other" stem does not establish that the sound is a synth.
    # Keep unidentified accompaniment visible without inventing a specific name.
    if not families and activity:
        families = [{'family': 'Unidentified accompaniment', 'score': None,
                     'status': 'Accompaniment detected, but no reliable instrument name',
                     'evidence': activity}]
    result['stem_activity'] = activity
    result['placeholder_families'] = families
    return families, result


def instrument_names(families):
    if not families:
        return ['No confident instrument match - choose sound']
    return [family['family'] + (' (estimated)' if family['family'] != 'Unidentified accompaniment' else '')
            for family in families]


def write_instrument_guide(folder, names, classification):
    lines = ['INSTRUMENT NAMES', '', *names, '',
             'Names are audio-classifier estimates. Exact plugins and presets are not identified.',
             'Detection examines short passages; Instruments.json records scores and timestamps.',
             'Instrument_Placeholders.mid is a one-bar import helper with only the named tracks.',
             'Use Create one channel per track. Separate files are in Instrument_MIDI.',
             'Each track has one visible C5 starter note; delete or replace it after choosing a sound.',
             'No instrumental phrases have been transcribed.']
    (Path(folder)/'Instruments.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (Path(folder)/'Instruments.json').write_text(json.dumps(classification, indent=2), encoding='utf-8')


def refresh_instruments(export_dir, progress=None, cancel_event=None):
    """Recheck names in an existing export while retaining its audio and drums."""
    progress = progress or (lambda message: None)
    folder = Path(export_dir).expanduser().resolve()
    report_path = folder/'Report.json'
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    source = Path(report['input']).expanduser().resolve()
    if not source.is_file():
        raise ValueError('The original audio file has moved. Restore it to '+str(source)+' and recheck.')
    stems = {name: folder/'Stems'/(name+'.wav') for name in ('vocals','drums','bass','guitar','piano','other')
             if (folder/'Stems'/(name+'.wav')).is_file()}
    with tempfile.TemporaryDirectory(prefix='.instrument-recheck-', dir=folder) as temporary_name:
        temporary = Path(temporary_name)
        mix, sr = vocal.read_audio(source, temporary, cancel_event)
        progress('Checking instrument names in the original audio and saved accompaniment…')
        families, classification = detect_instrument_names(source, stems, mix, sr, progress, cancel_event)
        names = instrument_names(families)
        ready = temporary/'ready'
        ready.mkdir()
        bpm = float(report['config']['target_bpm'])
        write_placeholder_midis(ready, names, bpm, 4)
        write_instrument_guide(ready, names, classification)
        if (folder/'Instrument MIDI fix.txt').is_file():
            (ready/'Instrument MIDI fix.txt').write_text(
                'Import Instrument_Placeholders.mid with All tracks and Create one channel per track.\n'
                'This one-bar file has a C5 starter note per named instrument.\n'
                'Delete or replace the starter notes after choosing your sounds.\n'
                'Instrument names and recognition evidence are in Instruments.txt and Instruments.json.\n'
                'Unidentified accompaniment means sound was detected without a reliable instrument name.\n',
                encoding='utf-8')
        if (folder/'Instrument_Placeholders_With_Notes.mid').is_file():
            shutil.copy2(ready/'Instrument_Placeholders.mid', ready/'Instrument_Placeholders_With_Notes.mid')
        flp_warning = None
        try:
            placeholder_flp = _placeholder_flp(ready/'Instrument_Placeholders.flp', names, bpm)
        except Exception as exc:
            placeholder_flp = False
            (ready/'Instrument_Placeholders.flp').unlink(missing_ok=True)
            flp_warning = 'Optional instrument FLP could not be updated: '+str(exc)
        report['instrument_estimates'] = families
        report['instrument_classifier'] = classification.get('model')
        report['placeholder_midi'] = {'starter_note': 60, 'velocity': 64, 'length_beats': 1,
                                     'file_length_beats': 4, 'transcribed': False, 'individual_files': 'Instrument_MIDI'}
        old_prefixes = ('No accompaniment stem', 'Instrument families are estimates from separated',
                        'Empty MIDI tracks', 'Each instrument MIDI', 'Isolated-vocal mode: instrument detection')
        report['warnings'] = [warning for warning in report.get('warnings', []) if not warning.startswith(old_prefixes)]
        report['warnings'] += list(classification.get('warnings', []))
        if flp_warning:
            report['warnings'].append(flp_warning)
        report['warnings'].append('Instrument names are audio-classifier estimates; check uncertain labels by ear.')
        report['warnings'] = list(dict.fromkeys(report['warnings']))
        (ready/'Report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        vocal.check_cancel(cancel_event)
        if report_path.read_bytes() != original_report:
            raise ValueError('This export changed during the recheck. Try again after it finishes saving.')
        files = [path for path in ready.rglob('*') if path.is_file()]
        backup = folder/'Instrument name backups'/f'{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(2)}'
        backup.mkdir(parents=True)
        # Preserve the exact prior named tracks and metadata before committing.
        existing = [folder/path.relative_to(ready) for path in files]
        existing += list((folder/'Instrument_MIDI').glob('*.mid')) if (folder/'Instrument_MIDI').is_dir() else []
        obsolete_flp = folder/'Instrument_Placeholders.flp'
        remove_flp = not placeholder_flp and obsolete_flp.is_file()
        if remove_flp:
            existing.append(obsolete_flp)
        preserved = set()
        for path in existing:
            if path.is_file() and path not in preserved:
                relative = path.relative_to(folder)
                copy = backup/relative
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copy)
                preserved.add(path)
        vocal.check_cancel(cancel_event)
        if report_path.read_bytes() != original_report:
            raise ValueError('This export changed during the recheck. Try again after it finishes saving.')
        committed = []
        stale = []
        try:
            for path in files:
                destination = folder/path.relative_to(ready)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
                committed.append(destination)
            current_individuals = {path.relative_to(ready) for path in files if path.parent.name == 'Instrument_MIDI'}
            for path in list((folder/'Instrument_MIDI').glob('*.mid')):
                if path.relative_to(folder) not in current_individuals:
                    path.unlink()
                    stale.append(path)
            if remove_flp:
                obsolete_flp.unlink()
                stale.append(obsolete_flp)
        except Exception:
            for destination in committed+stale:
                old = backup/destination.relative_to(folder)
                if old.is_file():
                    shutil.copy2(old, destination)
                elif destination.exists():
                    destination.unlink()
            raise
    summary = 'Instrument names updated: '+', '.join(names)
    progress(summary)
    return {'output_dir': str(folder), 'vocal_path': str(folder/'Vocal_ALIGNED.wav'),
            'preview_path': str(folder/'Vocal_WITH_DRUMS.wav'), 'drum_midi_path': str(folder/'Drums_Combined.mid'),
            'summary': summary, 'warnings': report['warnings'], 'instrument_names': names,
            'placeholder_flp': placeholder_flp}


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _placeholder_flp(path, names, bpm):
    import setup_engine
    installations = setup_engine.discover_installations()
    if not installations or not names:
        return False
    channels = [dict(kind='sampler', name=name[:100], mixer=index+1, volume=78., pan=0.,
                     color='#64B5F6', sample_path='', preset_path='',
                     envelope=dict(enabled=False, attack=0, hold=0, decay=25, sustain=100, release=10))
                for index, name in enumerate(names)]
    setup_engine.build_project({'title': 'Estimated instrument placeholders', 'bpm': bpm, 'channels': channels}, path, installations[0])
    return True


def process_song(input_path, output_parent, config=None, progress=None, cancel_event=None):
    config = config or BeatConfig()
    config.validate()
    progress = progress or (lambda message: None)
    source = Path(input_path).expanduser().resolve()
    parent = Path(output_parent).expanduser().resolve()
    if not source.is_file():
        raise ValueError('Choose an existing song or vocal file.')
    parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    warnings = []
    with tempfile.TemporaryDirectory(prefix='.vocal-beat-', dir=parent) as name:
        temporary = Path(name)
        ready = temporary / 'ready'
        ready.mkdir()
        progress('Reading the song…')
        mix, sr = vocal.read_audio(source, temporary, cancel_event)
        if len(mix)/sr > 1200:
            raise ValueError('Choose a song shorter than 20 minutes.')
        if sr < 16000:
            raise ValueError('Use audio at 16 kHz or higher for reliable vocal analysis.')
        vocal.check_cancel(cancel_event)
        families = []
        stems = {}
        classification = {}
        if config.input_is_vocal:
            voice = mix.copy()
            beat_audio = mix
            warnings.append('Vocal-input mode skips stem separation. Supply Source BPM if the original tempo differs.')
        else:
            from stem_backend import separate_song
            decoded = temporary / 'song.wav'
            sf.write(decoded, mix, sr, subtype='FLOAT')
            stems = separate_song(decoded, ready/'Stems', progress=progress, cancel_event=cancel_event)
            if 'vocals' not in stems:
                raise ValueError('The separator did not return a vocal stem.')
            voice, vocal_sr = sf.read(stems['vocals'], dtype='float32', always_2d=True)
            if vocal_sr != sr:
                divisor = math.gcd(sr, vocal_sr)
                mix = signal.resample_poly(mix, vocal_sr//divisor, sr//divisor, axis=0).astype(np.float32)
                sr = vocal_sr
            drums_path = stems.get('drums')
            beat_audio, beat_sr = sf.read(drums_path, dtype='float32', always_2d=True) if drums_path else (mix, sr)
            if beat_sr != sr:
                divisor = math.gcd(beat_sr, sr)
                beat_audio = signal.resample_poly(beat_audio, sr//divisor, beat_sr//divisor, axis=0)
        families, classification = detect_instrument_names(source, stems, mix, sr, progress, cancel_event)
        warnings.extend(classification.get('warnings', []))
        warnings.append('Instrument names are audio-classifier estimates. Instruments.json records scores and passages; exact synths and presets are not identified.')
        if not np.isfinite(voice).all() or float(np.max(np.abs(voice), initial=0)) < 1e-6:
            raise ValueError('No audible vocal was found. Try another song or supply an isolated vocal.')
        # Separator padding must not lengthen the source timeline.
        voice = voice[:len(mix)]
        if len(voice) < len(mix):
            voice = np.pad(voice, ((0,len(mix)-len(voice)), (0,0)))
        sf.write(ready/'Vocal_EXTRACTED.wav', voice, sr, subtype='PCM_24')
        progress('Estimating source tempo and changes between musical sections…')
        if config.source_bpm is not None:
            source_bpm, offset, confidence = float(config.source_bpm), 0., None
        elif config.input_is_vocal:
            source_bpm, offset, confidence = float(config.target_bpm), 0., None
        else:
            source_bpm, offset, confidence = detect_tempo(beat_audio, sr, config.target_bpm)
            warnings.append(f'Original tempo estimated at {source_bpm:g} BPM. If the phrasing feels wrong, enter the known Source BPM and run again.')
        sections = detect_sections(mix, sr, source_bpm, config.section_bars, offset)
        vocal.check_cancel(cancel_event)
        ratio = source_bpm / config.target_bpm
        if abs(ratio-1) > 1e-7:
            progress(f'Matching {source_bpm:g} to {config.target_bpm:g} BPM before beat-hit alignment…')
            prepared = vocal.stretch(voice, round(len(voice)*ratio), sr, temporary, cancel_event)
        else:
            prepared = voice
        snap_config = vocal.SnapConfig(bpm=config.target_bpm, sensitivity=config.sensitivity,
                                       min_slice_ms=config.min_slice_ms)
        progress('Finding vocal attacks and building four drum patterns…')
        anchors, analysis = vocal.detect_attacks(prepared, sr, snap_config)
        patterns = make_patterns(config.style, config.seed)
        planned, hits, targets, total_beats = plan_arrangement(sections, anchors, sr, source_bpm, config, patterns)
        if len(anchors) and not np.isin(targets, beat_frames([h['beat'] for h in hits], sr, config.target_bpm)).all():
            raise ValueError('A vocal target is missing from the exported drum arrangement.')
        frames = max(int(beat_frames(total_beats, sr, config.target_bpm)), int(targets[-1])+len(prepared)-int(anchors[-1]))
        # The final tail must also fit a complete two-bar ending.
        if frames > int(beat_frames(total_beats, sr, config.target_bpm)):
            extra = math.ceil((frames/sr*config.target_bpm/60-total_beats)/8)*8
            total_beats += extra
            frames = int(beat_frames(total_beats, sr, config.target_bpm))
            warnings.append('The last vocal tail extends into a padded ending after the drums.')
        aligned = np.zeros((frames, prepared.shape[1]), dtype=np.float32)
        if anchors[0] > 0:
            prefix = prepared[:anchors[0]].copy()
            if targets[0] == 0:
                if float(np.max(np.abs(prefix), initial=0)) > .003:
                    raise ValueError('The opening pickup needs room. Set First bar to 2 or later.')
                prefix = prefix[:0]
            if len(prefix) > targets[0]:
                if targets[0] < round(.04*sr) and float(np.max(np.abs(prefix), initial=0)) > .003:
                    raise ValueError('The opening pickup needs room. Set First bar to 2 or later.')
                prefix = vocal.stretch(prefix, max(1,int(targets[0])), sr, temporary, cancel_event)
            end = int(targets[0])
            aligned[end-len(prefix):end] = prefix
        timing = []
        compressed_count = 0
        for index, (begin, destination) in enumerate(zip(anchors, targets)):
            vocal.check_cancel(cancel_event)
            end = int(anchors[index+1]) if index+1 < len(anchors) else len(prepared)
            capacity = int(targets[index+1]-destination) if index+1 < len(targets) else frames-int(destination)
            piece, compressed, attack, fade, protected = vocal.fit_slice(prepared[begin:end], capacity, sr, temporary, cancel_event)
            if protected and np.max(np.abs(piece[fade:fade+protected]-prepared[begin+fade:begin+fade+protected])) > 1e-6:
                raise ValueError('An attack changed during rendering; output was not saved.')
            aligned[destination:destination+len(piece)] = piece
            compressed_count += int(compressed)
            section_index = next(i for i,s in enumerate(planned)
                                 if beat_frames((s['start_bar']-1)*4, sr, config.target_bpm) <= destination
                                 < beat_frames((s['end_bar_exclusive']-1)*4, sr, config.target_bpm))
            timing.append({'slice': index+1, 'section': section_index+1,
                           'source_seconds': float(begin/sr/ratio), 'prepared_seconds': float(begin/sr),
                           'target_seconds': float(destination/sr), 'target_midi_tick': round(destination/sr*config.target_bpm/60*PPQ),
                           'output_frames': len(piece), 'compressed': bool(compressed)})
            if index % 20 == 0 or index+1 == len(anchors):
                progress(f'Aligning vocal attacks to actual drum hits: {index+1}/{len(anchors)}…')
        if not np.isfinite(aligned).all():
            raise ValueError('Rendered audio contains invalid samples.')
        aligned *= min(1., .97/max(float(np.max(np.abs(aligned))), 1e-9))
        progress('Writing the vocal, five drum lanes, section patterns and instrument placeholders…')
        sf.write(ready/'Vocal_ALIGNED.wav', aligned, sr, subtype='PCM_24')
        drums = render_drums(hits, frames, sr, config.target_bpm, cancel_event)
        sf.write(ready/'Drums_PREVIEW.wav', drums, sr, subtype='PCM_24')
        preview = np.repeat(aligned, 2, axis=1) if aligned.shape[1] == 1 else aligned.copy()
        preview += drums * .65
        preview *= min(1., .98/max(float(np.max(np.abs(preview))), 1e-9))
        sf.write(ready/'Vocal_WITH_DRUMS.wav', preview, sr, subtype='PCM_24')
        write_midi(ready/'Drums_Combined.mid', hits, config.target_bpm, total_beats, planned)
        write_midi(ready/'Drums_By_Lane.mid', hits, config.target_bpm, total_beats, planned, split=True)
        patterns_dir = ready/'Patterns'
        patterns_dir.mkdir()
        for name, pattern in patterns.items():
            write_midi(patterns_dir/(name.replace(' ', '_')+'.mid'), pattern, config.target_bpm, 8)
        for section in planned:
            start_beat = (section['start_bar']-1)*4
            local = [{**hit, 'beat': hit['beat']-start_beat} for hit in hits if hit['section'] == section['section']]
            write_midi(patterns_dir/f"Section_{section['section']:02}_{section['pattern'].replace(' ', '_')}.mid",
                       local, config.target_bpm, section['bars']*4)
        names = instrument_names(families)
        write_placeholder_midis(ready, names, config.target_bpm, total_beats, planned)
        placeholder_flp = False
        try:
            placeholder_flp = _placeholder_flp(ready/'Instrument_Placeholders.flp', names, config.target_bpm)
        except Exception as error:
            warnings.append(f'Optional FLP placeholders could not be created: {error}. The MIDI and text instrument list were saved.')
        warnings += ['Sections are estimates from bar-level sound changes, with regular phrase boundaries when unclear; they are not guaranteed verse/chorus labels.',
                     'Vocal slices are acoustic attacks, not a lyric transcription. Listen for missed syllables and separation artifacts.',
                     'Each instrument MIDI includes one C5 starter note so FL can import it. Delete or replace that note; it is not a transcribed melody. Individual files are in Instrument_MIDI.',
                     'Changing drum timing later in FL Studio does not automatically move this rendered vocal. This export matches the generated beat.']
        if any(s['max_local_shift_ms'] > 350 for s in planned):
            warnings.append('Some vocal attacks moved more than 350 ms within a section. Review those sections in Timing.csv and the preview.')
        _csv(ready/'Timing.csv', timing)
        _csv(ready/'Sections.csv', [{k:v for k,v in section.items() if k != 'energy'} for section in planned])
        _csv(ready/'Drum_Hits.csv', [{'beat': h['beat'], 'seconds': h['beat']*60/config.target_bpm,
                                    'note': h['note'], 'lane': LANES[h['lane']], 'velocity': h['velocity'],
                                    'section': h['section']} for h in hits])
        write_instrument_guide(ready, names, classification)
        summary = f'{config.target_bpm:g} BPM · {len(planned)} sections · 4 beat patterns · {len(anchors)} vocal attacks aligned'
        report = {'config': asdict(config), 'input': str(source), 'source_bpm': source_bpm,
                  'tempo_confidence': confidence, 'sample_rate': sr, 'sections': planned,
                  'instrument_estimates': families, 'drum_note_map': dict(zip(LANES, NOTES)),
                  'instrument_classifier': classification.get('model'),
                  'placeholder_midi': {'starter_note': 60, 'velocity': 64, 'length_beats': 1,
                                       'file_length_beats': 4, 'transcribed': False, 'individual_files': 'Instrument_MIDI'},
                  'vocal_attacks': len(anchors), 'compressed_slices': compressed_count,
                  'all_targets_are_drum_hits': True, 'listening_verified': False,
                  'analysis': analysis, 'elapsed_seconds': round(time.time()-started,2),
                  'summary': summary, 'warnings': warnings}
        (ready/'Report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (ready/'FL Studio - Start Here.txt').write_text(
            f'VOCAL BEAT BUILDER — {config.target_bpm:g} BPM, 4/4\n\n'
            '1. Set FL Studio to the BPM above.\n'
            '2. Put Vocal_ALIGNED.wav at Playlist bar 1, including its leading silence.\n'
            '3. Import Drums_Combined.mid into one drum instrument mapped as follows:\n'
            '   C5 kick, C#5 snare, D5 off-snare, D#5 closed hat, E5 open hat.\n'
            '   These are MIDI notes 60, 61, 62, 63, 64, not General MIDI drum mapping.\n'
            '   Put this MIDI pattern at Playlist bar 1 too.\n'
            '4. Drums_By_Lane.mid is the alternative with five named drum tracks.\n'
            '   Patterns contains four 2-bar patterns plus exact section MIDI, including\n'
            '   any extra hats added for dense vocal delivery.\n'
            '5. Listen to Vocal_WITH_DRUMS.wav for a preview with simple synthesized drums.\n'
            '   Replace those drum sounds with your own samples in FL Studio.\n'
            '6. Import Instrument_Placeholders.mid with Create one channel per track.\n'
            '   Each instrument has one C5 starter note. Delete or replace it after import.\n'
            '   Instrument_MIDI contains separate files to drop onto individual instruments.\n'
            '   These notes are placeholders, not recovered melodies. The optional FLP has\n'
            '   blank named Sampler channels instead. Replace them with your synths.\n'
            '   Opening this FLP starts a separate project; save your current work first.\n\n'
            'Sections.csv lists exact one-based FL bar numbers. Timing.csv lists vocal\n'
            'attacks and corresponding drum MIDI ticks. Stems retains the source separation.\n'
            'Section and instrument labels are estimates. Preview the result by ear.\n'
            'Editing drum timing afterwards does not move the exported vocal automatically.\n', encoding='utf-8')
        vocal.check_cancel(cancel_event)
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', source.stem)[:65].strip(' .') or 'Song'
        dest = parent/f'{safe}_{config.target_bpm:g}BPM_{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(3)}'
        os.rename(ready, dest)
    progress('Finished. Listen to the vocal with its new drum beat.')
    return {'output_dir': str(dest), 'vocal_path': str(dest/'Vocal_ALIGNED.wav'),
            'preview_path': str(dest/'Vocal_WITH_DRUMS.wav'), 'drum_midi_path': str(dest/'Drums_Combined.mid'),
            'summary': summary, 'warnings': warnings, 'placeholder_flp': placeholder_flp, 'instrument_names': names}

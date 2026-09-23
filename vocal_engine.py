"""Vocal Grid Snap: local onset slicing and hard, pitch-preserving quantization.

Detection locates acoustic attacks, not linguistically guaranteed word boundaries.
Each attack is moved intact; only a slice body that would overlap is compressed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

ROOT = Path(__file__).resolve().parent
GRIDS = {'1/8': 2, '1/16': 4, '1/32': 8, '1/8T': 3, '1/16T': 6}


@dataclass
class SnapConfig:
    bpm: float = 140.0
    grid: str = '1/16'
    sensitivity: float = .6
    min_slice_ms: float = 80.0
    offset_ms: float = 0.0
    source_bpm: float | None = None
    export_slices: bool = False
    section_mode: str = 'off'
    section_gap_beats: float = 2.0
    section_block_bars: int = 2
    start_bar: int = 1

    def validate(self):
        for name in ('bpm', 'sensitivity', 'min_slice_ms', 'offset_ms'):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f'{name} must be a finite number.')
        if not 30 <= self.bpm <= 300:
            raise ValueError('BPM must be between 30 and 300.')
        if self.grid not in GRIDS:
            raise ValueError('Choose a supported grid.')
        if not 0 <= self.sensitivity <= 1:
            raise ValueError('Sensitivity must be between 0 and 100%.')
        if not 40 <= self.min_slice_ms <= 500:
            raise ValueError('Minimum slice must be 40 to 500 ms.')
        if abs(self.offset_ms) > 60000:
            raise ValueError('Beat offset must be within 60 seconds of file start.')
        if self.source_bpm is not None and (
                not math.isfinite(self.source_bpm) or not 30 <= self.source_bpm <= 300):
            raise ValueError('Original BPM must be blank or between 30 and 300.')
        if self.section_mode not in ('off', 'blocks'):
            raise ValueError('Section mode must be off or blocks.')
        if not math.isfinite(self.section_gap_beats) or not .25 <= self.section_gap_beats <= 32:
            raise ValueError('New section silence must be between 0.25 and 32 beats.')
        if self.section_block_bars not in (1, 2, 4):
            raise ValueError('Section block size must be 1, 2, or 4 bars.')
        if not isinstance(self.start_bar, int) or not 1 <= self.start_bar <= 999:
            raise ValueError('First section bar must be a whole number from 1 to 999.')


def check_cancel(event):
    if event is not None and event.is_set():
        raise RuntimeError('Cancelled')


def run_tool(command, event):
    check_cancel(event)
    flags = 0
    if os.name == 'nt':
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, creationflags=flags,
                          env=environment) as child:
        while True:
            try:
                out, err = child.communicate(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                if event is not None and event.is_set():
                    child.terminate()
                    child.communicate()
                    raise RuntimeError('Cancelled')
        if child.returncode:
            raise RuntimeError(err.decode(errors='replace')[-2000:] or 'Audio tool failed.')


def read_audio(path, temporary, event):
    try:
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        ffmpeg = ROOT / 'tools' / 'ffmpeg.exe'
        if not ffmpeg.is_file():
            raise ValueError('Use a WAV/FLAC/AIFF file, or restore tools/ffmpeg.exe.')
        decoded = temporary / 'decoded.wav'
        run_tool([ffmpeg, '-v', 'error', '-nostdin', '-y', '-i', path,
                  '-vn', '-c:a', 'pcm_f32le', decoded], event)
        audio, sr = sf.read(str(decoded), dtype='float32', always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('The audio is empty or contains invalid samples.')
    if audio.shape[1] > 2:
        raise ValueError('Please export a mono or stereo isolated vocal stem.')
    if np.max(np.abs(audio)) < 1e-6:
        raise ValueError('This file is silent. Choose a vocal stem.')
    return audio, sr


def stretch(audio, frames, sr, temporary, event):
    """Fixed-ratio offline stretching, never resampling to change pitch."""
    frames = max(1, int(frames))
    if frames == len(audio):
        return audio.copy()
    if np.max(np.abs(audio), initial=0) < 1e-8:
        return np.zeros((frames, audio.shape[1]), dtype=np.float32)
    exe = ROOT / 'tools' / 'rubberband-r3.exe'
    if not exe.is_file():
        raise ValueError('Missing tools/rubberband-r3.exe. Extract the whole tool folder.')
    source = temporary / 'slice_input.wav'
    target = temporary / 'slice_output.wav'
    sf.write(source, audio, sr, subtype='FLOAT')
    run_tool([exe, '-3', '--centre-focus', '-t', f'{frames / len(audio):.12f}',
              '-q', source, target], event)
    result, result_sr = sf.read(target, dtype='float32', always_2d=True)
    if result_sr != sr or result.shape[1] != audio.shape[1] or not np.isfinite(result).all():
        raise ValueError('Time-stretcher returned invalid audio.')
    # Fixed-ratio offline mode has duration rounding at the final frame.
    if abs(len(result) - frames) > max(8, int(.005 * sr)):
        raise ValueError('Unexpected time-stretcher length; no output has been finalized.')
    if len(result) < frames:
        result = np.pad(result, ((0, frames-len(result)), (0, 0)))
    return result[:frames]


def detect_attacks(audio, sr, config):
    """Positive spectral change plus energy rises, including starts after silence."""
    # Use the stronger channel so out-of-phase stereo does not disappear in analysis.
    channel = int(np.argmax(np.mean(audio.astype(np.float64)**2, axis=0)))
    analysis_sr = 22050
    divisor = math.gcd(sr, analysis_sr)
    mono = signal.resample_poly(audio[:, channel], analysis_sr//divisor, sr//divisor)
    hop = 110  # About 5 ms.
    window = 512
    if len(mono) < window:
        return np.array([0], dtype=np.int64), {'candidates': 1}
    _, times, spectrum = signal.stft(mono, fs=analysis_sr, nperseg=window,
                                    noverlap=window-hop, boundary='zeros', padded=True)
    magnitude = np.abs(spectrum)
    bands = magnitude[2:210]
    log_mag = np.log1p(bands*500)
    flux = np.maximum(np.diff(log_mag, axis=1, prepend=log_mag[:, :1]), 0).mean(axis=0)
    energy_sample = ndimage.uniform_filter1d(mono.astype(np.float64)**2, size=221)
    frame_pos = np.minimum(np.rint(times*analysis_sr).astype(int), len(mono)-1)
    rms = np.sqrt(np.maximum(energy_sample[frame_pos], 0))
    rms = ndimage.gaussian_filter1d(rms, .7)
    floor = max(float(np.quantile(rms, .98))*.012, 1e-6)
    log_energy = np.log(np.maximum(rms, floor))
    rise = np.maximum(np.diff(log_energy, prepend=log_energy[0]), 0)
    def normalize(x):
        positive = x[x > 1e-9]
        denominator = np.quantile(positive, .9) if len(positive) else 1.
        return np.clip(x/max(float(denominator), 1e-9), 0, 4)
    score = .72*normalize(flux) + .28*normalize(rise)
    score = ndimage.gaussian_filter1d(score, .6)
    local = ndimage.median_filter(score, size=31)
    novelty = np.maximum(score-.7*local, 0)
    spacing = max(1, round(config.min_slice_ms/1000*analysis_sr/hop))
    threshold = .46 - .34*config.sensitivity
    peaks, _ = signal.find_peaks(novelty, distance=spacing, prominence=threshold,
                                 height=threshold)
    active = rms > floor
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    # Only keep starts after a genuine quiet gap, not a tiny dip inside a vowel.
    gap_frames = max(1, round(.065*analysis_sr/hop))
    starts = [p for p in starts if p == 0 or not np.any(active[max(0,p-gap_frames):p])]
    candidates = []
    starts_set = set(starts)
    # Refine starts in sample time; centered analysis windows otherwise put the
    # slice a few milliseconds before a sharp consonant, making that consonant late.
    sample_level = np.sqrt(np.maximum(ndimage.uniform_filter1d(
        audio[:, channel].astype(np.float64)**2, size=max(3, round(sr*.001))), 0))
    for p in sorted(set(peaks.tolist()+starts)):
        if not np.any(active[p:min(len(active), p+5)]):
            continue
        before = float(np.mean(rms[max(0, p-5):p+1]))
        after = float(np.mean(rms[p+2:min(len(rms), p+9)])) if p+2 < len(rms) else 0.
        if p not in starts_set and after < before*.42:
            continue  # A broadband vocal release is not a new syllable attack.
        lo = max(0, p-5)
        # The preceding energy minimum marks the front of the acoustic attack.
        q = lo + int(np.argmin(rms[lo:p+1]))
        if p in starts_set:
            q = p
        sample = min(len(audio)-1, max(0, round(times[q]*sr)))
        refine_start = max(0, sample-round(.007*sr))
        refine_end = min(len(audio), round((times[p]+.022)*sr))
        if refine_end > refine_start+4:
            segment = sample_level[refine_start:refine_end]
            valley_end = min(len(segment), max(1, sample-refine_start+1))
            valley_index = int(np.argmin(segment[:valley_end]))
            baseline = float(segment[valley_index])
            high = float(np.max(segment[valley_index:]))
            if high > max(baseline*1.6, floor*2):
                threshold_level = max(floor, baseline+.08*(high-baseline))
                crossings = np.flatnonzero(segment[valley_index:] >= threshold_level)
                if len(crossings):
                    sample = refine_start+valley_index+int(crossings[0])
        if sample > len(audio)-round(.04*sr):
            continue  # Ignore the analysis-window boundary; retain this tail in the last slice.
        importance = float(novelty[p]) + (3.0 if p in starts_set else 0.0)
        candidates.append((sample, importance))
    # A stem or excerpt can begin in an already sounding syllable. Preserve
    # that beginning as its own anchor instead of refining past existing audio.
    initial_level = float(np.sqrt(np.mean(audio[:max(1, round(.002*sr)), channel]**2)))
    if initial_level > floor:
        candidates.append((0, 20.0))
    # Retain strongest candidate when multiple detections describe the same attack.
    selected = []
    min_samples = round(config.min_slice_ms*sr/1000)
    for sample, strength in sorted(candidates, key=lambda p: p[1], reverse=True):
        if all(abs(sample-other) >= min_samples for other in selected):
            selected.append(sample)
    if not selected:
        raise ValueError('No vocal attacks detected. Raise sensitivity or use a cleaner vocal stem.')
    return np.asarray(sorted(selected), dtype=np.int64), {
        'candidates': len(candidates), 'analysis_hop_ms': hop/analysis_sr*1000,
        'detector': 'spectral change + energy rise; acoustic syllable candidates'}


def ordered_grid(anchors, sr, config, minimum_target_frame=0):
    """Least-square ordered placement on distinct cells using pooled adjacent violators."""
    step = 60/config.bpm/GRIDS[config.grid]
    offset = config.offset_ms/1000
    raw = (anchors/sr-offset)/step
    observations = raw - np.arange(len(raw))
    blocks = []
    for value in observations:
        blocks.append([float(value), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            count = left[1]+right[1]
            blocks.append([(left[0]*left[1]+right[0]*right[1])/count, count])
    fitted = np.concatenate([np.repeat(mean, count) for mean, count in blocks])
    minimum_cell = math.ceil((minimum_target_frame/sr-offset)/step-1e-10)
    cells = np.maximum(np.floor(fitted+.5).astype(np.int64), minimum_cell) + np.arange(len(raw))
    targets = np.rint((offset+cells*step)*sr).astype(np.int64)
    shifts = (targets-anchors)/sr
    if np.max(np.abs(shifts)) > max(.16, step*1.5):
        raise ValueError('Too many detected syllables compete for this grid. Choose a finer grid '
                         '(1/32), lower sensitivity, or increase minimum slice. No syllables were discarded.')
    if np.any(np.diff(targets) <= 0) or targets[0] < 0:
        raise ValueError('Could not create an ordered beat grid.')
    return targets, cells, step


def fit_slice(piece, capacity, sr, temporary, event):
    """Keep consonant attack unchanged; compress body only when it cannot fit."""
    original_length = len(piece)
    result = piece.copy()
    compressed = capacity is not None and original_length > capacity
    attack = min(round(.020*sr), original_length, capacity if capacity is not None else original_length)
    if compressed:
        if capacity <= attack+round(.015*sr):
            raise ValueError('Grid slot is too short for this attack. Use a coarser grid '
                             '(such as 1/16) and reduce detection sensitivity if needed.')
        body = stretch(piece[attack:], capacity-attack, sr, temporary, event)
        result = np.concatenate([piece[:attack], body], axis=0)
        # Smooth only the join into the stretched body; the attack stays intact.
        cross = min(round(.002*sr), len(body))
        if cross > 1:
            ramp = np.linspace(0, 1, cross, dtype=np.float32)[:, None]
            continuation = piece[attack:attack+cross]
            result[attack:attack+cross] = continuation*(1-ramp)+body[:cross]*ramp
    fade = min(round(.0015*sr), len(result)//4)
    if fade:
        result[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        result[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    protected = max(0, min(attack, len(result)-fade)-fade)
    return result, compressed, attack, fade, protected


def make_click(frames, sr, channels, config):
    result = np.zeros((frames, channels), dtype=np.float32)
    beat = 60/config.bpm
    offset = config.offset_ms/1000
    for beat_index in range(math.ceil(-offset/beat), math.ceil((frames/sr-offset)/beat)):
        start = round((offset+beat_index*beat)*sr)
        count = min(round(.023*sr), frames-start)
        if start < 0 or count <= 0:
            continue
        t = np.arange(count)/sr
        accent = beat_index % 4 == 0
        pulse = (.17 if accent else .11)*np.sin(2*np.pi*(1700 if accent else 1100)*t)*np.exp(-190*t)
        result[start:start+count] += pulse[:, None]
    return result


def arrange_sections(audio, sr, config, anchors=None):
    """Group rendered vocal by quiet gaps and move intact groups to 4/4 bar blocks.

    Anchor-based translation preserves the internal syllable grid. A separate,
    much lower threshold retains faint tails and pre-attack pickups. This is
    pause detection, not verse/chorus recognition.
    """
    anchors = np.asarray(anchors if anchors is not None else [0], dtype=np.int64)
    if config.section_mode == 'off':
        return audio, [], anchors.copy()
    config.validate()
    if not len(anchors) or np.any(np.diff(anchors) <= 0) or anchors[0] < 0 or anchors[-1] >= len(audio):
        raise ValueError('Section placement needs valid ordered vocal attack positions.')
    hop = max(1, round(sr*.005))
    amplitude = np.max(np.abs(audio), axis=1)
    power = amplitude.astype(np.float64)**2
    framed = np.pad(power, (0, (-len(power)) % hop)).reshape(-1, hop)
    rms = np.sqrt(framed.mean(axis=1))
    nonzero = rms[rms > 1e-8]
    if not len(nonzero):
        raise ValueError('The snapped vocal is silent; no sections can be arranged.')
    quiet_threshold = max(1e-6, float(np.quantile(nonzero, .95))*.01)
    tail_threshold = max(2e-6, float(amplitude.max())*.0001)
    quiet = rms < quiet_threshold
    transitions = np.diff(np.r_[False, quiet, False].astype(np.int8))
    gap_starts = np.flatnonzero(transitions == 1)*hop
    gap_ends = np.minimum(np.flatnonzero(transitions == -1)*hop, len(audio))
    gap_required = config.section_gap_beats*60/config.bpm*sr
    cuts = [0]
    for begin, end in zip(gap_starts, gap_ends):
        if end-begin >= gap_required and begin > anchors[0] and end <= anchors[-1]:
            # A faint reverb tail can be below the section-detection threshold.
            # Split only within real near-silence so that tail never becomes a
            # pickup attached to the next section.
            very_quiet = amplitude[begin:end] <= tail_threshold
            edges = np.diff(np.r_[False, very_quiet, False].astype(np.int8))
            silence_starts = np.flatnonzero(edges == 1)
            silence_ends = np.flatnonzero(edges == -1)
            if not len(silence_starts):
                continue
            lengths = silence_ends-silence_starts
            longest = int(np.argmax(lengths))
            if lengths[longest] < max(1, round(sr*.005)):
                continue
            middle = int(begin+(silence_starts[longest]+silence_ends[longest])//2)
            if np.any((anchors >= cuts[-1]) & (anchors < middle)) and np.any(anchors >= middle):
                cuts.append(middle)
    cuts.append(len(audio))
    bar_frames = sr*60/config.bpm*4  # Explicit 4/4.
    offset_frames = config.offset_ms/1000*sr
    block = config.section_block_bars
    next_bar = config.start_bar
    mapped = np.zeros_like(anchors)
    rows = []
    pieces = []
    previous_end = 0
    for left, right in zip(cuts, cuts[1:]):
        indices = np.flatnonzero((anchors >= left) & (anchors < right))
        if not len(indices):
            continue
        first_anchor = int(anchors[indices[0]])
        last_anchor = int(anchors[indices[-1]])
        sound = np.flatnonzero(amplitude[left:right] > tail_threshold)
        # Keep all attacks even if one falls in a very quiet breath.
        start = min(first_anchor, int(left+sound[0]) if len(sound) else first_anchor)
        end = max(last_anchor+1, int(left+sound[-1]+1) if len(sound) else last_anchor+1)
        preroll = first_anchor-start
        target_anchor = round(offset_frames+(next_bar-1)*bar_frames)
        # A pickup may use the prior block's empty space, but never its vocal
        # tail or negative time. Advance even the first block until it fits.
        while target_anchor-preroll < previous_end:
            next_bar += block
            target_anchor = round(offset_frames+(next_bar-1)*bar_frames)
        duration_after_anchor = end-first_anchor
        reserved_bars = max(block, math.ceil((duration_after_anchor-.51)/(bar_frames*block))*block)
        # Source and destination rounding can differ by one sample at fractional BPM.
        # Absorb that one sample within the 2 ms end fade, not an entire extra block.
        reserved_end = round(offset_frames+(next_bar-1+reserved_bars)*bar_frames)
        excess = target_anchor+duration_after_anchor-reserved_end
        if excess == 1 and duration_after_anchor > round(.01*sr):
            end -= 1
            duration_after_anchor -= 1
        elif excess > 0:
            reserved_bars += block
            reserved_end = round(offset_frames+(next_bar-1+reserved_bars)*bar_frames)
        target_start = target_anchor-preroll
        target_end = target_anchor+duration_after_anchor
        piece = audio[start:end].copy()
        # Boundaries are in almost-silence. Avoid creating clicks at the new joins.
        fade = min(round(.002*sr), len(piece)//4)
        if fade and start < first_anchor:
            piece[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        if fade:
            piece[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
        pieces.append((target_start, piece))
        mapped[indices] = anchors[indices]+target_anchor-first_anchor
        rows.append(dict(section=len(rows)+1, source_start_frame=int(start), source_end_frame=int(end),
                         source_anchor_frame=first_anchor, target_start_frame=int(target_start),
                         target_end_frame=int(target_end), target_anchor_frame=int(target_anchor),
                         source_start_seconds=start/sr, source_end_seconds=end/sr,
                         target_start_seconds=target_start/sr, target_end_seconds=target_end/sr,
                         target_anchor_seconds=target_anchor/sr, start_bar=int(next_bar),
                         vocal_bars=(end-first_anchor)/bar_frames, reserved_bars=int(reserved_bars),
                         next_start_bar=int(next_bar+reserved_bars),
                         shift_ms=(target_anchor-first_anchor)/sr*1000))
        previous_end = target_end
        next_bar += reserved_bars
    output_frames = max(previous_end, round(offset_frames+(next_bar-1)*bar_frames))
    arranged = np.zeros((output_frames, audio.shape[1]), dtype=np.float32)
    for start, piece in pieces:
        arranged[start:start+len(piece)] = piece
    return arranged, rows, mapped


def process(input_path, output_parent, config=None, progress=None, cancel_event=None):
    config = config or SnapConfig()
    config.validate()
    progress = progress or (lambda message: None)
    source = Path(input_path).expanduser().resolve()
    parent = Path(output_parent).expanduser().resolve()
    if not source.is_file():
        raise ValueError('Choose an existing vocal audio file.')
    check_cancel(cancel_event)
    started = time.time()
    progress('Reading vocal. All processing stays on this computer.')
    parent.mkdir(parents=True, exist_ok=True)
    # Everything remains in a unique scratch folder until the whole render succeeds.
    with tempfile.TemporaryDirectory(prefix='.vocal-snap-', dir=parent) as temporary_name:
        temporary = Path(temporary_name)
        audio, sr = read_audio(source, temporary, cancel_event)
        source_frames = len(audio)
        if config.source_bpm and abs(config.source_bpm-config.bpm) > 1e-7:
            progress('Matching the original tempo to your project BPM without changing pitch…')
            audio = stretch(audio, round(len(audio)*config.source_bpm/config.bpm), sr, temporary, cancel_event)
        progress('Finding syllable-sized attacks throughout the vocal…')
        anchors, detection = detect_attacks(audio, sr, config)
        check_cancel(cancel_event)
        targets, cells, step = ordered_grid(anchors, sr, config)
        pickup_grid_adjusted = False
        if targets[0] == 0 and anchors[0] > 0 and np.max(np.abs(audio[:anchors[0]]), initial=0) > .003:
            # Close-trimmed vocals often have a soft attack before the detector's
            # anchor. Keep that attack by using the next available grid position.
            targets, cells, step = ordered_grid(anchors, sr, config, minimum_target_frame=int(anchors[0]))
            pickup_grid_adjusted = True
        progress(f'{len(anchors)} attacks found. Snapping every attack fully to {config.grid} notes…')
        output_frames = max(len(audio), int(targets[-1])+len(audio)-int(anchors[-1]))
        snapped = np.zeros((output_frames, audio.shape[1]), dtype=np.float32)
        staged = temporary/'ready'
        staged.mkdir()
        slices_dir = staged/'Slices'
        if config.export_slices:
            slices_dir.mkdir()
        # Preserve audio before the first anchor, including leading silence and pickups.
        if anchors[0] and targets[0]:
            prefix = audio[:anchors[0]]
            if len(prefix) > targets[0]:
                prefix = stretch(prefix, int(targets[0]), sr, temporary, cancel_event)
            position = max(0, int(targets[0])-len(prefix))
            fade = min(round(.0015*sr), len(prefix))
            prefix = prefix.copy()
            if fade:
                prefix[-fade:] *= np.linspace(1, 0, fade)[:, None]
            snapped[position:position+len(prefix)] = prefix
        elif anchors[0] > 0 and targets[0] == 0:
            # Do not silently discard audible pre-roll when snapping the first hit to zero.
            if np.max(np.abs(audio[:anchors[0]]), initial=0) > .003:
                raise ValueError('The first hit would cut audible pre-roll at file start. '
                                 'Add a little leading silence or a positive beat offset.')
        rows = []
        rendered_check = 0.
        compressed_count = 0
        last_update = 0.
        for index, (begin, destination) in enumerate(zip(anchors, targets)):
            check_cancel(cancel_event)
            end = int(anchors[index+1]) if index+1 < len(anchors) else len(audio)
            capacity = int(targets[index+1]-destination) if index+1 < len(targets) else None
            piece, compressed, attack, fade, protected = fit_slice(
                audio[begin:end], capacity, sr, temporary, cancel_event)
            snapped[destination:destination+len(piece)] = piece
            if protected:
                error = np.max(np.abs(piece[fade:fade+protected]-audio[begin+fade:begin+fade+protected]))
                rendered_check = max(rendered_check, float(error))
            compressed_count += int(compressed)
            name = f'{index+1:04d}_at_{destination/sr:010.5f}s.wav'
            if config.export_slices and config.section_mode == 'off':
                sf.write(slices_dir/name, piece, sr, subtype='FLOAT')
            rows.append(dict(slice=index+1, source_start_seconds=float(begin/sr),
                             source_end_seconds=end/sr, target_start_seconds=float(destination/sr),
                             grid_cell=int(cells[index]), shift_ms=float((destination-begin)/sr*1000),
                             source_frames=int(end-begin), output_frames=len(piece),
                             compressed=bool(compressed), attack_preserved_frames=protected,
                             slice_file=f'Slices/{name}' if config.export_slices else ''))
            now = time.time()
            if now-last_update > 1.5 or index == len(anchors)-1:
                progress(f'Placed {index+1}/{len(anchors)} syllable slices on the grid.')
                last_update = now
        if not np.isfinite(snapped).all() or rendered_check > 1e-6:
            raise ValueError('Rendered attack verification failed; output was not finalized.')
        section_rows = []
        final_targets = targets.copy()
        if config.section_mode == 'blocks':
            check_cancel(cancel_event)
            progress('Detecting vocal sections and placing them in non-overlapping bar blocks…')
            snapped, section_rows, final_targets = arrange_sections(snapped, sr, config, targets)
            section_starts = np.asarray([s['source_anchor_frame'] for s in section_rows])
            for index, row in enumerate(rows):
                section_index = int(np.searchsorted(section_starts, targets[index], side='right')-1)
                section = section_rows[section_index]
                row['section'] = section_index+1
                row['syllable_target_start_seconds'] = row['target_start_seconds']
                row['target_start_seconds'] = float(final_targets[index]/sr)
                row['section_shift_ms'] = float((final_targets[index]-targets[index])/sr*1000)
                row['total_shift_ms'] = float((final_targets[index]-anchors[index])/sr*1000)
                row['grid_cell'] = int(round((final_targets[index]/sr-config.offset_ms/1000)/step))
                row['output_frames'] = min(row['output_frames'], section['source_end_frame']-int(targets[index]))
                if config.export_slices:
                    name = f'{index+1:04d}_at_{final_targets[index]/sr:010.5f}s.wav'
                    row['slice_file'] = f'Slices/{name}'
                    begin = int(final_targets[index])
                    sf.write(slices_dir/name, snapped[begin:begin+row['output_frames']], sr, subtype='FLOAT')
            sections_dir = staged/'Sections'
            sections_dir.mkdir()
            for section in section_rows:
                start, end = section['target_start_frame'], section['target_end_frame']
                name = f"{section['section']:02d}_FLbar_{section['start_bar']:03d}_{section['reserved_bars']}bars.wav"
                # Pad each file to its reserved end, without copying another section's pickup.
                block_end = round((config.offset_ms/1000+(section['next_start_bar']-1)*4*60/config.bpm)*sr)
                section_audio = np.zeros((max(end, block_end)-start, snapped.shape[1]), dtype=np.float32)
                section_audio[:end-start] = snapped[start:end]
                sf.write(sections_dir/name, section_audio, sr, subtype='FLOAT')
                section['file'] = f'Sections/{name}'
            with (staged/'Sections.csv').open('w', newline='', encoding='utf-8-sig') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(section_rows[0]))
                writer.writeheader()
                writer.writerows(section_rows)
            progress(f"{len(section_rows)} sections placed in {config.section_block_bars}-bar blocks; no overlaps.")
        progress('Writing the FL Studio vocal, metronome, and timing report…')
        peak = float(np.max(np.abs(snapped)))
        gain = min(1., .98/max(peak, 1e-9))
        snapped *= gain
        sf.write(staged/'Vocal_SNAPPED.wav', snapped, sr, subtype='PCM_24')
        click = make_click(len(snapped), sr, audio.shape[1], config)
        sf.write(staged/'Metronome.wav', click, sr, subtype='PCM_24')
        practice = snapped+click
        practice *= min(1., .98/max(float(np.max(np.abs(practice))), 1e-9))
        sf.write(staged/'Snapped_WITH_CLICK.wav', practice, sr, subtype='PCM_24')
        original = audio.copy()
        original *= gain
        original += make_click(len(original), sr, audio.shape[1], config)
        original *= min(1., .98/max(float(np.max(np.abs(original))), 1e-9))
        sf.write(staged/'Original_WITH_CLICK.wav', original, sr, subtype='PCM_24')
        with (staged/'Timing.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        warnings = ['Automatic cuts locate acoustic attacks; check the result by ear for missed or extra syllables.']
        if pickup_grid_adjusted:
            warnings.append('The first hit used the next grid position to preserve its soft attack before the detected anchor.')
        if section_rows:
            warnings.append('Sections are detected from quiet gaps in 4/4. This rearranges pauses; it does not identify verse/chorus labels.')
            if section_rows[0]['start_bar'] != config.start_bar:
                warnings.append(f"The opening pickup needs more space: the first section anchor starts at FL bar {section_rows[0]['start_bar']} (earliest requested bar {config.start_bar}).")
            if len(section_rows) == 1:
                warnings.append('Only one section found. Reduce New section after silence to split shorter pauses.')
        if np.max(np.abs(targets-anchors))/sr > .15:
            warnings.append('Some syllables moved more than 150 ms. Try 1/32 if the rhythm feels rearranged.')
        report = dict(config=asdict(config), source=str(source), sample_rate=sr,
                      channels=audio.shape[1], source_duration_seconds=source_frames/sr,
                      prepared_duration_seconds=len(audio)/sr, output_duration_seconds=len(snapped)/sr,
                      slices=len(rows), compressed_slices=compressed_count,
                      sections=len(section_rows), section_block_bars=config.section_block_bars,
                      section_map=section_rows, time_signature='4/4',
                      max_shift_ms=float(np.max(np.abs(targets-anchors))/sr*1000),
                      median_shift_ms=float(np.median(np.abs(targets-anchors))/sr*1000),
                      grid_step_ms=step*1000, master_gain=gain,
                      attack_copy_max_error=rendered_check,
                      grid_sample_rounding_error_ms=float(np.max(np.abs(targets/sr-(config.offset_ms/1000+cells*step)))*1000),
                      final_grid_sample_rounding_error_ms=float(np.max(np.abs(
                          (final_targets/sr-config.offset_ms/1000)/step-
                          np.rint((final_targets/sr-config.offset_ms/1000)/step)))*step*1000),
                      analysis=detection, elapsed_seconds=round(time.time()-started,2),
                      listening_verified=False, warnings=warnings)
        (staged/'Report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (staged/'FL Studio - Start Here.txt').write_text(
            f'Set FL Studio to {config.bpm:g} BPM.\n'
            'Drag Vocal_SNAPPED.wav into the Playlist at the project start (bar 1).\n'
            'Keep the file at its exported tempo; do not stretch or trim its leading silence.\n'
            'Use Metronome.wav at the same start to check alignment. It is separate from the vocal.\n'
            'Original_WITH_CLICK.wav and Snapped_WITH_CLICK.wav let you compare the timing.\n'
            'Timing.csv lists each source cut and destination. Automatic cuts need listening review.\n'
            'Audio attacks snap 100%; body compression preserves pitch where slices would collide.\n'
            'Individual slices, if enabled, are 32-bit float at unity gain; their positions are in Timing.csv.\n',
            encoding='utf-8')
        if section_rows:
            with (staged/'FL Studio - Start Here.txt').open('a', encoding='utf-8') as handle:
                handle.write('\nSECTION ARRANGEMENT (4/4)\n'
                             f"First section anchor: FL bar {section_rows[0]['start_bar']}. Blocks: {config.section_block_bars} bars.\n"
                             'Sections.csv lists final section start/end positions and reserved bars.\n'
                             'From FL bar 1, four bars of space means the next section starts at bar 5.\n'
                             'For literal even labels (2, 4, 6...), set First section bar to 2 and blocks to 2.\n'
                             'Whole sections are moved intact after syllable snapping; unused block space is silence.\n'
                             'Sections/ contains individual 32-bit float sections. A pickup can begin before its named anchor bar; use Sections.csv for its exact position.\n')
        check_cancel(cancel_event)
        stem = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in source.stem)[:70]
        run_name = f'{stem}_{config.bpm:g}BPM_{datetime.now():%Y%m%d_%H%M%S_%f}'
        destination_dir = parent/run_name
        staged.rename(destination_dir)
        report.update(output_dir=str(destination_dir), vocal_path=str(destination_dir/'Vocal_SNAPPED.wav'),
                      click_path=str(destination_dir/'Metronome.wav'),
                      comparison_path=str(destination_dir/'Snapped_WITH_CLICK.wav'),
                      report_path=str(destination_dir/'Report.json'))
        progress(f'Done: {len(rows)} slices snapped. Originals are unchanged.')
        return report


def main():
    parser = argparse.ArgumentParser(description='Hard-quantize vocal syllable attacks for FL Studio.')
    parser.add_argument('input', type=Path)
    parser.add_argument('--bpm', type=float, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--grid', choices=list(GRIDS), default='1/16')
    parser.add_argument('--sensitivity', type=float, default=.6)
    parser.add_argument('--min-slice-ms', type=float, default=80.)
    parser.add_argument('--offset-ms', type=float, default=0.)
    parser.add_argument('--source-bpm', type=float)
    parser.add_argument('--slices', action='store_true')
    parser.add_argument('--sections', action='store_true', help='Arrange vocal sections in 4/4 bar blocks.')
    parser.add_argument('--section-gap-beats', type=float, default=2.)
    parser.add_argument('--block-bars', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--start-bar', type=int, default=1)
    args = parser.parse_args()
    config = SnapConfig(args.bpm, args.grid, args.sensitivity, args.min_slice_ms,
                        args.offset_ms, args.source_bpm, args.slices,
                        section_mode='blocks' if args.sections else 'off',
                        section_gap_beats=args.section_gap_beats,
                        section_block_bars=args.block_bars, start_bar=args.start_bar)
    result = process(args.input, args.output or args.input.parent/'Vocal Grid Snap Exports',
                     config, progress=lambda msg: print(msg, flush=True))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

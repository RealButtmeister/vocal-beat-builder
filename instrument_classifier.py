"""Local PANNs instrument classification, separate from stem-energy estimates.

Public imports use only the standard library. Audio and model inference stay
local, in an owned cancellable subprocess using the app's private CPU runtime.
"""
from __future__ import annotations

import csv
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request

from stem_backend import (
    APP_DIR, MODEL_DIR, RUNTIME_DIR, StemSeparationError, _check_cancel,
    _notify, _run_process, ensure_runtime,
)

MODEL_FILENAME = "MobileNetV1_mAP=0.389.pth"
MODEL_URL = "https://zenodo.org/records/3987831/files/MobileNetV1_mAP%3D0.389.pth?download=1"
MODEL_MD5 = "a419303e1c88aa1b9d2ac3811563d371"
MODEL_BYTES = 23639473
TORCHLIBROSA_VERSION = "0.1.0"
_LOCK = threading.Lock()

# Indices follow the authors' AudioSet class CSV, rather than filenames/stems.
FAMILY_CLASSES = {
    "bass": (142, 194),
    "guitar": (140, 141, 143, 144),
    "piano": (153, 154),
    "keys": (152, 155, 156, 157, 160),
    "synth": (158, 159, 213),
    "strings": (189, 190, 191, 192, 193),
    "brass": (185, 186, 187, 188),
    "woodwind": (195, 196, 197, 198, 208, 210, 211),
    "plucked_strings": (139, 147, 148, 149, 150, 151, 199),
    "mallet_percussion": (178, 179, 180, 181, 182, 183),
    "accordion": (209,),
}
FAMILY_NAMES = {
    "bass": "Bass", "guitar": "Guitar", "piano": "Piano", "keys": "Keys / organ",
    "synth": "Synthesizer", "strings": "Bowed strings", "brass": "Brass",
    "woodwind": "Woodwind", "plucked_strings": "Plucked strings",
    "mallet_percussion": "Mallet instruments", "accordion": "Accordion",
}


def ensure_classifier_runtime(progress=None, cancel_event=None) -> Path:
    python = ensure_runtime(progress, cancel_event)
    marker = RUNTIME_DIR / "vbb-classifier-ready.json"
    while not _LOCK.acquire(timeout=0.2):
        _check_cancel(cancel_event)
    try:
        _check_cancel(cancel_event)
        if marker.is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")).get("torchlibrosa") == TORCHLIBROSA_VERSION:
                    return python
            except (OSError, ValueError):
                pass
        _notify(progress, "Preparing instrument recognition (a small one-time audio helper)…")
        log = APP_DIR / "Classifier setup log.txt"
        _run_process([str(python), "-m", "pip", "install", "--no-deps",
                      "torchlibrosa==" + TORCHLIBROSA_VERSION], log, progress,
                     cancel_event, "Preparing instrument recognition")
        _run_process([str(python), "-c",
                      "from torchlibrosa.stft import Spectrogram, LogmelFilterBank; "
                      "from importlib.metadata import version; "
                      "assert version('torchlibrosa') == '0.1.0'"], log,
                     progress, cancel_event, "Checking instrument recognition")
        _check_cancel(cancel_event)
        marker.write_text(json.dumps({"torchlibrosa": TORCHLIBROSA_VERSION}), encoding="utf-8")
        return python
    finally:
        _LOCK.release()


def classify_instruments(input_path, progress=None, cancel_event=None,
                         accompaniment_paths=None) -> dict:
    """Classify the actual source audio; optional extra audio never replaces it.

    Scores are model outputs, not calibrated probabilities of instrument presence.
    Optional accompaniment_paths may be a sequence of paths or a name/path dict.
    """
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise StemSeparationError(f"Cannot find audio for instrument recognition: {source}")
    extras = []
    if accompaniment_paths:
        values = accompaniment_paths.values() if isinstance(accompaniment_paths, dict) else accompaniment_paths
        extras = [str(Path(p).expanduser().resolve()) for p in values]
        for item in extras:
            if not Path(item).is_file():
                raise StemSeparationError(f"Cannot find accompaniment audio: {item}")
    python = ensure_classifier_runtime(progress, cancel_event)
    _notify(progress, "Listening for instruments in the original audio…")
    with tempfile.TemporaryDirectory(prefix="vbb-classify-") as temporary:
        request = Path(temporary) / "request.json"
        result = Path(temporary) / "result.json"
        request.write_text(json.dumps({"source": str(source), "extras": extras}), encoding="utf-8")
        _run_process([str(python), str(Path(__file__).resolve()), "--worker",
                      str(request), str(result)], APP_DIR / "Classifier log.txt",
                     progress, cancel_event, "Recognizing instruments")
        _check_cancel(cancel_event)
        if not result.is_file():
            raise StemSeparationError("Instrument recognition finished without a result. See Classifier log.txt.")
        return json.loads(result.read_text(encoding="utf-8"))


def _status(message):
    print("VBB_STATUS:" + message, flush=True)


def _checksum(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_path():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / MODEL_FILENAME
    if target.is_file() and _checksum(target) == MODEL_MD5:
        return target
    _status("First run: downloading the 24 MB instrument recognition model. Your audio stays here.")
    descriptor, temporary = tempfile.mkstemp(prefix="panns-", suffix=".part", dir=MODEL_DIR)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "VocalBeatBuilder/1"})
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0"))
            received = 0
            next_report = 5 * 1024 * 1024
            while block := response.read(1024 * 1024):
                handle.write(block)
                received += len(block)
                if received >= next_report:
                    _status(f"Instrument model download: {received / 1000000:.0f} MB" +
                            (f" of {total / 1000000:.0f} MB" if total else ""))
                    next_report += 5 * 1024 * 1024
        if _checksum(temporary) != MODEL_MD5:
            raise RuntimeError("The instrument model download was incomplete or did not match the author's checksum. Try again.")
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _summarize(windows, labels):
    """Conservative display rules, not a calibrated instrument-presence test.

    One stronger match or repeated moderate matches can produce an estimate.
    Original/accompaniment views of the same instant count only once. Broad
    parent labels must not create duplicate tracks for a supported child label.
    """
    candidates = []
    for family, indices in FAMILY_CLASSES.items():
        evidence = []
        for window in windows:
            scores = window["instrument_scores"]
            index = max(indices, key=lambda i: scores[str(i)])
            evidence.append({"label": labels[index], "start_seconds": window["start_seconds"],
                             "end_seconds": window["end_seconds"], "score": scores[str(index)],
                             "source": window["source"]})
        evidence.sort(key=lambda item: item["score"], reverse=True)
        candidates.append({"family": family, "name": FAMILY_NAMES[family],
                           "score": evidence[0]["score"] if evidence else 0.0,
                           "evidence": evidence[:8], "status": "uncertain"})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    for candidate in candidates:
        temporal_hits = {item["start_seconds"] for item in candidate["evidence"] if item["score"] >= .25}
        if candidate["score"] >= .35 or len(temporal_hits) >= 2:
            selected.append(candidate)
    selected_ids = {item["family"] for item in selected}
    families = []
    for candidate in selected:
        identifier = candidate["family"]
        best_label = candidate["evidence"][0]["label"]
        if identifier == "keys" and best_label == "Keyboard (musical)" and selected_ids & {"piano", "synth"}:
            continue
        if identifier == "plucked_strings" and best_label == "Plucked string instrument" and selected_ids & {"guitar", "bass"}:
            continue
        family = dict(candidate)
        family.update(family=candidate["name"], family_id=identifier, status="estimated")
        family["evidence"] = [item for item in candidate["evidence"] if item["score"] >= .25]
        families.append(family)
    return families, candidates


def _worker(request_path, result_path):
    import math
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from panns_model import MobileNetV1

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checkpoint = _model_path()
    _status("Loading the instrument recognition model…")
    model = MobileNetV1().cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    with (APP_DIR / "panns_class_labels.csv").open(encoding="utf-8", newline="") as handle:
        labels = [row["display_name"] for row in csv.DictReader(handle)]
    if len(labels) != 527:
        raise RuntimeError("Instrument class labels are missing or incomplete. Restore the app's panns_class_labels.csv.")
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    windows = []
    skipped_windows = 0
    phase_safe_reads = 0
    all_indices = sorted({index for indices in FAMILY_CLASSES.values() for index in indices})
    with ExitStack() as stack:
        decoded_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="vbb-classify-audio-")))

        def open_audio(path):
            try:
                return stack.enter_context(sf.SoundFile(path))
            except (OSError, RuntimeError):
                ffmpeg = APP_DIR / "tools" / "ffmpeg.exe"
                executable = str(ffmpeg) if ffmpeg.is_file() else shutil.which("ffmpeg")
                if not executable:
                    raise RuntimeError("This audio format needs FFmpeg. Restore the app's tools folder.")
                decoded = decoded_dir / (str(len(list(decoded_dir.iterdir()))) + ".wav")
                completed = subprocess.run(
                    [executable, "-hide_banner", "-loglevel", "error", "-nostdin",
                     "-i", str(path), "-vn", "-ar", "32000", "-ac", "1",
                     "-c:a", "pcm_f32le", str(decoded)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    check=False)
                if completed.returncode:
                    raise RuntimeError("The audio could not be read for instrument recognition: " +
                                       completed.stderr.decode("utf-8", "replace")[-1200:])
                return stack.enter_context(sf.SoundFile(decoded))

        original = open_audio(request["source"])
        extra_handles = [open_audio(path) for path in request.get("extras", [])]
        sample_rate, frames = original.samplerate, len(original)
        duration = frames / sample_rate
        if not frames:
            raise RuntimeError("The source audio is empty.")
        for handle in extra_handles:
            if abs(len(handle) / handle.samplerate - duration) > .1:
                raise RuntimeError("Accompaniment audio must share the source's timing. Use this song's original separated stems.")
        # Final anchored window includes the tail without creating tiny trailing windows.
        last_start = max(0.0, duration - 10.0)
        starts = list(np.arange(0.0, last_start, 5.0)) + [last_start]
        def read_window(handle, start, end):
            nonlocal phase_safe_reads
            rate = handle.samplerate
            offset, size = round(start * rate), round((end - start) * rate)
            handle.seek(min(offset, len(handle)))
            channels = handle.read(min(size, len(handle) - handle.tell()), dtype="float32", always_2d=True)
            if not np.isfinite(channels).all():
                raise RuntimeError("The source audio contains invalid samples.")
            audio = channels.mean(axis=1)
            if len(channels) and channels.shape[1] > 1:
                energies = np.mean(channels.astype(np.float64) ** 2, axis=0)
                strongest = int(np.argmax(energies))
                if np.mean(audio.astype(np.float64) ** 2) < energies[strongest] * .1:
                    audio = channels[:, strongest]
                    phase_safe_reads += 1
            audio = np.pad(audio, (0, size - len(audio)))
            if rate != 32000:
                divisor = math.gcd(rate, 32000)
                audio = resample_poly(audio, 32000 // divisor, rate // divisor).astype(np.float32)
            target = round((end - start) * 32000)
            return np.pad(audio[:target], (0, max(0, target - len(audio))))

        for source_name in (["original", "accompaniment"] if extra_handles else ["original"]):
            for number, start in enumerate(starts, 1):
                end = min(duration, float(start) + 10.0)
                mix_audio = read_window(original, float(start), end)
                mix_rms = float(np.sqrt(np.mean(mix_audio.astype(np.float64) ** 2)))
                audio = mix_audio.copy()
                gain = 1.0
                if source_name == "accompaniment":
                    audio = sum((read_window(handle, float(start), end) for handle in extra_handles), np.zeros_like(mix_audio))
                audio_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                if not np.isfinite(audio).all():
                    raise RuntimeError("The source audio contains invalid samples.")
                # Gate per window, not whole-song average. Do not amplify tiny
                # separation residue until it appears to be a real instrument.
                if audio_rms < (max(1e-5, mix_rms * .025) if source_name == "accompaniment" else 1e-6):
                    skipped_windows += 1
                    continue
                gain = min(8.0, max(1.0, .05 / audio_rms))
                audio *= gain
                if len(audio) < 32000:
                    audio = np.pad(audio, (0, 32000 - len(audio)))
                _status(f"Recognizing instruments: window {number} of {len(starts)} ({source_name})")
                with torch.inference_mode():
                    scores = model(torch.from_numpy(audio).unsqueeze(0))[0].numpy()
                if scores.shape != (527,) or not np.isfinite(scores).all():
                    raise RuntimeError("The instrument model returned invalid scores.")
                indices = np.argsort(scores)[-20:][::-1]
                windows.append({
                    "source": source_name, "start_seconds": round(float(start), 4),
                    "end_seconds": round(min(duration, float(start) + 10.0), 4),
                    "source_rms": audio_rms, "mix_rms": mix_rms, "analysis_gain": gain,
                    "top_labels": [{"label": labels[i], "score": round(float(scores[i]), 6)} for i in indices],
                    "instrument_scores": {str(i): round(float(scores[i]), 6) for i in all_indices},
                })
    families, candidates = _summarize(windows, labels)
    result = {
        "families": families, "candidates": candidates, "windows": windows,
        "skipped_quiet_windows": skipped_windows,
        "phase_safe_channel_reads": phase_safe_reads,
        "selection": {"single_window_score": .35, "repeated_window_score": .25,
                      "repeated_window_count": 2, "calibrated": False},
        "model": {"name": "PANNs MobileNetV1", "dataset": "AudioSet", "labels": 527,
                  "checkpoint": MODEL_FILENAME, "md5": MODEL_MD5, "weights_license": "CC-BY-4.0",
                  "reference": "https://zenodo.org/records/3987831", "score_type": "uncalibrated model score"},
        "warnings": ["Instrument names are estimates from audio. Model scores are not certainty percentages; quiet or blended instruments may be missed."]}
    Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("Run instrument classification from Vocal Beat Builder.")

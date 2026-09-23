"""Local, cancellable six-stem separation in a private Python environment.

Audio stays on this computer. First use downloads the runtime and a Demucs
checkpoint. This module itself uses only the standard library so the GUI can
import it before the optional separation dependencies have been installed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Callable

APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = APP_DIR / ".runtime"
MODEL_DIR = APP_DIR / "models"
MODEL_NAME = "htdemucs_6s.yaml"
STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
SEPARATOR_VERSION = "0.47.0"
TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
SETUP_VERSION = 2
_SETUP_LOCK = threading.Lock()
Progress = Callable[[str], None] | None


class StemSeparationError(RuntimeError):
    """An actionable separation or installation error."""


class StemCancelled(StemSeparationError):
    """The user cancelled an owned separation/setup process."""


def _notify(progress: Progress, message: str) -> None:
    if progress is not None:
        progress(message)


def _cancelled(cancel_event) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _check_cancel(cancel_event) -> None:
    if _cancelled(cancel_event):
        raise StemCancelled("Cancelled. Your source audio has not been changed.")


def _python_path(runtime: Path = RUNTIME_DIR) -> Path:
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "2",
        # This first release intentionally uses the CPU to avoid competing
        # with games or FL Studio for graphics memory.
        "CUDA_VISIBLE_DEVICES": "-1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    })
    env["PATH"] = str(APP_DIR / "tools") + os.pathsep + env.get("PATH", "")
    return env


def _stop_owned_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # PID comes only from our Popen instance. /T also stops a pip build
        # child; it does not stop unrelated Python/FL Studio processes.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW, check=False,
        )
    else:
        import signal
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=4)


def _run_process(command: list[str], log_path: Path, progress: Progress = None,
                 cancel_event=None, stage: str = "Working") -> None:
    """Drain output off-thread and poll cancellation even during silent work."""
    _check_cancel(cancel_event)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    messages: queue.Queue[str | None] = queue.Queue()
    tail: list[str] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nStage: " + stage + "\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, env=_environment(), creationflags=flags,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise StemSeparationError(f"Could not start {stage.lower()}: {exc}") from exc

        def read_output() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        finished_reading = False
        last_status = time.monotonic()
        try:
            while process.poll() is None or not finished_reading:
                _check_cancel(cancel_event)
                try:
                    line = messages.get(timeout=0.15)
                except queue.Empty:
                    if time.monotonic() - last_status > 20:
                        _notify(progress, stage + " — still working…")
                        last_status = time.monotonic()
                    continue
                if line is None:
                    finished_reading = True
                    continue
                log.write(line)
                log.flush()
                tail.append(line.strip())
                tail = tail[-24:]
                if line.startswith("VBB_STATUS:"):
                    _notify(progress, line.split(":", 1)[1].strip())
                    last_status = time.monotonic()
            code = process.wait()
        except BaseException:
            _stop_owned_process(process)
            raise
        finally:
            reader.join(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
        if code:
            detail = "\n".join(tail)[-3500:]
            raise StemSeparationError(
                f"{stage} failed. Details were saved to {log_path}.\n{detail}"
            )


def runtime_ready() -> bool:
    marker = RUNTIME_DIR / "vbb-ready.json"
    if not _python_path().is_file() or not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return (data.get("setup_version") == SETUP_VERSION
                and data.get("audio_separator") == SEPARATOR_VERSION
                and data.get("torch") == TORCH_VERSION
                and data.get("torchvision") == TORCHVISION_VERSION)
    except (OSError, ValueError):
        return False


def ensure_runtime(progress: Progress = None, cancel_event=None) -> Path:
    """Install the CPU backend once into this app's .runtime, if necessary.

    Call from a GUI worker thread. Cancellation stops only our installer tree.
    Failed/partial installs are resumed on the next call; a ready marker is
    written only after a real import check succeeds.
    """
    while not _SETUP_LOCK.acquire(timeout=0.2):
        _check_cancel(cancel_event)
    try:
        _check_cancel(cancel_event)
        if runtime_ready():
            return _python_path()
        log_path = APP_DIR / "Setup log.txt"
        _notify(progress, "First run: setting up local vocal separation. This downloads its tools once.")
        python_path = _python_path()
        if not python_path.is_file():
            _run_process([sys.executable, "-m", "venv", str(RUNTIME_DIR)], log_path,
                         progress, cancel_event, "Creating the private audio runtime")
        _run_process([str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
                     log_path, progress, cancel_event, "Preparing the installer")
        _notify(progress, "Downloading the CPU audio engine. Your song is not uploaded.")
        _run_process([
            str(python_path), "-m", "pip", "install", f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ], log_path, progress, cancel_event, "Installing the CPU audio engine")
        _notify(progress, "Installing the stem separator and audio readers…")
        _run_process([
            str(python_path), "-m", "pip", "install",
            f"audio-separator[cpu]=={SEPARATOR_VERSION}",
            f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
            "audioread==3.0.1",
        ], log_path, progress, cancel_event, "Installing local stem separation")
        _run_process([
            str(python_path), "-c",
            "import torch, torchvision, soundfile; from audio_separator.separator import Separator; "
            "from importlib.metadata import version; "
            f"assert version('audio-separator') == '{SEPARATOR_VERSION}'; "
            f"assert torch.__version__.split('+')[0] == '{TORCH_VERSION}'; "
            f"assert torchvision.__version__.split('+')[0] == '{TORCHVISION_VERSION}'; "
            "print('audio-separator', version('audio-separator')); print('torch', torch.__version__)",
        ], log_path, progress, cancel_event, "Checking the audio engine")
        _check_cancel(cancel_event)
        (RUNTIME_DIR / "vbb-ready.json").write_text(json.dumps({
            "setup_version": SETUP_VERSION, "audio_separator": SEPARATOR_VERSION,
            "torch": TORCH_VERSION, "torchvision": TORCHVISION_VERSION, "device": "cpu",
        }, indent=2), encoding="utf-8")
        _notify(progress, "The local audio engine is ready.")
        return python_path
    except StemCancelled:
        raise
    except StemSeparationError as exc:
        raise StemSeparationError(
            "The local vocal separator could not finish its first-time setup. "
            "Check your internet connection and free disk space, then try again. "
            "Your existing Python tools were not changed.\n" + str(exc)
        ) from exc
    finally:
        _SETUP_LOCK.release()


setup_dependencies = ensure_runtime


def separate_song(input_path, output_dir, progress: Progress = None,
                  cancel_event=None) -> dict[str, Path]:
    """Return six genuine Demucs stem paths; never substitute the full mix."""
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise StemSeparationError(f"The source audio could not be found: {source}")
    output.mkdir(parents=True, exist_ok=True)
    for name in STEMS:
        if (output / f"{name}.wav").exists():
            raise StemSeparationError("This output folder already contains stems. Choose a fresh export folder.")
    python_path = ensure_runtime(progress, cancel_event)
    if not shutil.which("ffmpeg", path=_environment().get("PATH")):
        raise StemSeparationError(
            "FFmpeg is missing. Keep the app's tools folder next to stem_backend.py, "
            "then restart the app."
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log_path = output / "Separation log.txt"
    _notify(progress, "Separating vocals, drums, bass, guitar, piano and other sounds locally…")
    with tempfile.TemporaryDirectory(prefix=".stem-run-", dir=output) as work_name:
        work = Path(work_name)
        result_path = work / "result.json"
        _run_process([
            str(python_path), str(Path(__file__).resolve()), "--worker",
            str(source), str(work), str(MODEL_DIR), str(result_path),
        ], log_path, progress, cancel_event, "Separating the song")
        _check_cancel(cancel_event)
        try:
            report = json.loads(result_path.read_text(encoding="utf-8"))
            worker_paths = {name: Path(report["stems"][name]).resolve() for name in STEMS}
            for name, path in worker_paths.items():
                if not path.is_relative_to(work.resolve()) or not path.is_file():
                    raise ValueError(f"Missing or invalid {name} stem")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise StemSeparationError(
                "The separator did not produce all six valid stems. "
                f"See {log_path}. No full-mix substitute was exported."
            ) from exc
        created: list[Path] = []
        try:
            outputs: dict[str, Path] = {}
            for name, path in worker_paths.items():
                _check_cancel(cancel_event)
                destination = output / f"{name}.wav"
                with destination.open("xb") as dst, path.open("rb") as src:
                    created.append(destination)
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                outputs[name] = destination
            (output / "Stem estimates.json").write_text(json.dumps({
                "model": MODEL_NAME,
                "note": "These are energy estimates of separated families, not exact instrument or plugin identification. "
                        "Guitar and piano can contain bleed; absence/presence is uncertain.",
                "families": report["families"],
            }, indent=2), encoding="utf-8")
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
    _notify(progress, "Vocal and instrument stems are ready.")
    return outputs


def _worker(source: Path, output: Path, models: Path, result: Path) -> None:
    # Imports happen only in our owned worker; the GUI keeps responding.
    import logging
    import numpy as np
    import soundfile as sf
    import torch
    from audio_separator.separator import Separator

    class AtomicDownloadSeparator(Separator):
        def download_file_if_not_exists(self, url, output_path):
            """A cancelled download must never masquerade as a cached model."""
            destination = Path(output_path)
            if destination.is_file():
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".download-", suffix=".part", dir=destination.parent,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            # Upstream skips existing files, so reserve a unique name and
            # remove the zero-byte reservation before asking it to download.
            temporary.unlink()
            try:
                super().download_file_if_not_exists(url, str(temporary))
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise StemSeparationError("The model download was empty. Check the connection and retry.")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    info = sf.info(str(source))
    if info.frames <= 0 or info.channels not in (1, 2):
        raise StemSeparationError("Use a non-empty mono or stereo audio file.")
    mix_energy_sum = 0.0
    mix_samples = 0
    for block in sf.blocks(str(source), blocksize=262144, dtype="float32", always_2d=True):
        if not np.isfinite(block).all():
            raise StemSeparationError("The source contains invalid audio samples.")
        mix_energy_sum += float(np.sum(np.square(block, dtype=np.float64)))
        mix_samples += block.size
    mix_energy = mix_energy_sum / max(1, mix_samples)
    print("VBB_STATUS:Loading the local six-stem model; first use downloads its weights.", flush=True)
    separator = AtomicDownloadSeparator(
        model_file_dir=str(models), output_dir=str(output), output_format="WAV",
        use_soundfile=True, normalization_threshold=1.0, amplification_threshold=0.0,
        log_level=logging.INFO,
        demucs_params={"segment_size": "7", "shifts": 1,
                       "overlap": 0.25, "segments_enabled": True},
    )
    separator.load_model(model_filename=MODEL_NAME)
    print("VBB_STATUS:Extracting the vocal and five instrument families on the CPU…", flush=True)
    separator.separate(str(source), custom_output_names={name: name for name in STEMS})
    paths: dict[str, str] = {}
    energies: dict[str, float] = {}
    expected_duration = info.frames / info.samplerate
    for name in STEMS:
        path = output / f"{name}.wav"
        if not path.is_file():
            # Some separator versions match custom names using title case.
            candidates = [p for p in output.glob("*.wav")
                          if p.stem.lower() == name or f"({name})" in p.stem.lower()]
            if len(candidates) != 1:
                raise StemSeparationError(f"The model did not output an identifiable {name} stem.")
            path = candidates[0]
        stem_info = sf.info(str(path))
        if stem_info.frames <= 0 or abs(stem_info.duration - expected_duration) > 0.1:
            raise StemSeparationError(f"The {name} stem has an unexpected length; separation was not accepted.")
        energy_sum = 0.0
        sample_count = 0
        for block in sf.blocks(str(path), blocksize=262144, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise StemSeparationError(f"The {name} stem contains invalid audio samples.")
            energy_sum += float(np.sum(np.square(block, dtype=np.float64)))
            sample_count += block.size
        energies[name] = energy_sum / max(1, sample_count)
        paths[name] = str(path.resolve())
    total_instrument_energy = sum(energies[name] for name in STEMS if name != "vocals")
    families = {name: {
        "rms": energies[name] ** 0.5,
        "instrument_energy_share": (energies[name] / total_instrument_energy
                                    if total_instrument_energy and name != "vocals" else None),
        "energy_relative_to_mix": energies[name] / mix_energy if mix_energy else 0.0,
        "estimated_only": True,
    } for name in STEMS}
    result.write_text(json.dumps({"stems": paths, "families": families}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        if len(sys.argv) == 6 and sys.argv[1] == "--worker":
            _worker(*(Path(value).resolve() for value in sys.argv[2:]))
        elif sys.argv[1:] == ["--setup"]:
            ensure_runtime(progress=lambda message: print(message, flush=True))
        else:
            raise SystemExit("Use --setup to install the local separation engine.")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

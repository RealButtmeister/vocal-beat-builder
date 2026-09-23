"""Optional first-run installer for Vocal Beat Builder's local audio engine."""
from stem_backend import ensure_runtime

if __name__ == "__main__":
    ensure_runtime(progress=lambda message: print(message, flush=True))

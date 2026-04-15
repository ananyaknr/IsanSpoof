import librosa
import numpy as np

TARGET_SR = 16000
SEGMENT_DURATION = 4.0
SEGMENT_SAMPLES = int(TARGET_SR * SEGMENT_DURATION)   # 64,000 samples


def load_and_normalize(filepath: str) -> np.ndarray:
    audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def pad_or_trim(audio: np.ndarray, length: int = SEGMENT_SAMPLES) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    else:
        repeats = (length // len(audio)) + 1
        return np.tile(audio, repeats)[:length]


def preprocess_file(filepath: str) -> np.ndarray:
    """Load, normalize, and pad/trim a single audio file to a fixed length array."""
    audio = load_and_normalize(filepath)
    audio = pad_or_trim(audio)
    return audio  # shape: (64000,)

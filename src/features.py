import numpy as np
import librosa
from scipy.fftpack import dct

# ── LFCC configuration (matching ASVspoof 2019 organizer defaults) ──
LFCC_CONFIG = {
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 400,
    'n_filter': 70,
    'n_lfcc': 60,
    'sr': 16000,
    'fmin': 0.0,
    'fmax': 8000.0,
}

MFCC_CONFIG = {
    'n_mfcc': 60,
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 400,
    'n_mels': 128,
    'sr': 16000,
}

CQCC_CONFIG = {
    'sr': 16000,
    'fmin': librosa.note_to_hz('C1'),
    'n_bins': 72,
    'bins_per_octave': 12,
    'hop_length': 160,
    'n_cqcc': 60,
}


def linear_filterbank(sr, n_fft, n_filter, fmin, fmax):
    """Build a linear-spaced triangular filterbank matrix."""
    freqs = np.linspace(fmin, fmax, n_filter + 2)
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_filter, len(fft_freqs)))
    for i in range(n_filter):
        low, center, high = freqs[i], freqs[i + 1], freqs[i + 2]
        up_slope   = (fft_freqs - low)   / (center - low + 1e-8)
        down_slope = (high - fft_freqs) / (high - center + 1e-8)
        fb[i] = np.maximum(0, np.minimum(up_slope, down_slope))
    return fb


def extract_lfcc(audio: np.ndarray, config: dict = LFCC_CONFIG) -> np.ndarray:
    """Extract LFCC + delta + delta-delta. Returns shape (T, 180)."""
    stft = np.abs(librosa.stft(
        audio,
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        win_length=config['win_length'],
        window='hann'
    ))
    fb = linear_filterbank(config['sr'], config['n_fft'],
                           config['n_filter'], config['fmin'], config['fmax'])
    filter_energies = np.dot(fb, stft)
    log_energies = np.log(filter_energies + 1e-8)
    lfcc = dct(log_energies, type=2, axis=0, norm='ortho')
    lfcc = lfcc[:config['n_lfcc'], :]
    delta1 = librosa.feature.delta(lfcc, order=1)
    delta2 = librosa.feature.delta(lfcc, order=2)
    return np.concatenate([lfcc, delta1, delta2], axis=0).T  # (T, 180)


def extract_mfcc(audio: np.ndarray, config: dict = MFCC_CONFIG) -> np.ndarray:
    """Extract MFCC + delta + delta-delta. Returns shape (T, 180)."""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=config['sr'],
        n_mfcc=config['n_mfcc'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        win_length=config['win_length'],
        n_mels=config['n_mels']
    )
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta1, delta2], axis=0).T  # (T, 180)


def extract_cqcc(audio: np.ndarray, config: dict = CQCC_CONFIG) -> np.ndarray:
    """Extract CQCC + delta + delta-delta. Returns shape (T, 180)."""
    cqt = np.abs(librosa.cqt(
        audio,
        sr=config['sr'],
        fmin=config['fmin'],
        n_bins=config['n_bins'],
        bins_per_octave=config['bins_per_octave'],
        hop_length=config['hop_length']
    ))
    log_cqt = np.log(cqt + 1e-8)
    cqcc = dct(log_cqt, type=2, axis=0, norm='ortho')
    cqcc = cqcc[:config['n_cqcc'], :]
    delta1 = librosa.feature.delta(cqcc, order=1)
    delta2 = librosa.feature.delta(cqcc, order=2)
    return np.concatenate([cqcc, delta1, delta2], axis=0).T  # (T, 180)

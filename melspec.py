#!/usr/bin/env python3
"""
Mel spectrogram reference implementation.
Matches librosa.feature.melspectrogram(y=y, sr=sr, center=False, htk=True).
Uses only numpy and scipy (no audio libraries).
"""
import argparse
import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft


def hz_to_mel(f):
    """HTK mel scale: mel = 2595 * log10(1 + f/700)"""
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    """Inverse HTK mel scale: f = 700 * (10^(m/2595) - 1)"""
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filterbank(sr, n_fft, n_mels=128, f_min=0.0, f_max=None):
    """
    Build HTK mel filterbank with Slaney area normalization.
    Matches librosa.filters.mel(sr, n_fft, htk=True) -- verified zero diff.

    Parameters
    ----------
    sr : int
        Sample rate in Hz.
    n_fft : int
        FFT size (must be power of 2).
    n_mels : int
        Number of Mel bands.
    f_min : float
        Lowest frequency in Hz.
    f_max : float or None
        Highest frequency in Hz. Defaults to sr/2.

    Returns
    -------
    weights : np.ndarray, shape (n_mels, n_fft//2+1), dtype float32
        Mel filterbank matrix.
    """
    if f_max is None:
        f_max = sr / 2.0

    # FFT bin center frequencies
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)  # shape: (n_fft//2+1,)

    # n_mels+2 equally-spaced mel points (including left and right edges)
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)  # back to Hz

    # Build triangular filters
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f_l = hz_points[m]
        f_c = hz_points[m + 1]
        f_r = hz_points[m + 2]

        rising = (fft_freqs - f_l) / (f_c - f_l)
        falling = (f_r - fft_freqs) / (f_r - f_c)
        weights[m] = np.maximum(0.0, np.minimum(rising, falling)).astype(np.float32)

        # Slaney area normalization: divide by filter bandwidth in Hz
        enorm = 2.0 / (f_r - f_l)
        weights[m] *= enorm

    return weights  # shape: (n_mels, n_fft//2+1)


def compute_melspectrogram(y, sr=44100, n_fft=2048, hop_length=512,
                            n_mels=128, f_min=0.0, f_max=None):
    """
    Compute mel spectrogram from audio signal.
    Matches librosa.feature.melspectrogram(y=y, sr=sr, center=False, htk=True).

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Audio signal as float32 array.
    sr : int
        Sample rate in Hz.
    n_fft : int
        FFT size (must be power of 2).
    hop_length : int
        Number of samples between successive frames.
    n_mels : int
        Number of Mel bands.
    f_min : float
        Lowest frequency in Hz.
    f_max : float or None
        Highest frequency in Hz. Defaults to sr/2.

    Returns
    -------
    S : np.ndarray, shape (n_mels, n_frames), dtype float32
        Mel spectrogram.
    """
    if f_max is None:
        f_max = sr / 2.0

    # Step 1: Periodic Hann window -- fftbins=True is MANDATORY
    # This gives the periodic form used by librosa, NOT the symmetric np.hanning()
    window = get_window('hann', n_fft, fftbins=True)  # shape: (n_fft,), float64

    # Step 2: Frame the signal -- center=False, no zero-padding at edges
    n_samples = len(y)
    n_frames = 1 + (n_samples - n_fft) // hop_length
    shape = (n_frames, n_fft)
    strides = (y.strides[0] * hop_length, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    # NOTE: as_strided returns a read-only view; do not modify in-place

    # Step 3: Apply window (creates new array; promotes to float64 since window is float64)
    windowed = frames * window  # (n_frames, n_fft), float64

    # Step 4: Real FFT (unnormalized forward transform), power spectrum
    X = rfft(windowed, axis=1, n=n_fft)  # (n_frames, n_fft//2+1), complex128
    S = np.abs(X) ** 2                    # power spectrum, float64
    S = S.T                               # (n_fft//2+1, n_frames)

    # Step 5: Build HTK mel filterbank with Slaney area normalization
    mel_fb = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max)  # (n_mels, n_fft//2+1)

    # Step 6: Apply filterbank and cast to float32
    return (mel_fb @ S.astype(np.float32)).astype(np.float32)  # (n_mels, n_frames)


def main():
    parser = argparse.ArgumentParser(
        description='Compute mel spectrogram from PCM audio. '
                    'Matches librosa.feature.melspectrogram(center=False, htk=True).'
    )
    parser.add_argument('input_pcm', help='Input PCM file (int16 LE mono)')
    parser.add_argument('output_bin', help='Output binary file (float32 LE row-major [n_mels, n_frames])')
    parser.add_argument('--sr', type=int, default=44100,
                        help='Sample rate in Hz (default: 44100)')
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT size, must be power of 2 (default: 2048)')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Samples between successive frames (default: 512)')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='Number of Mel bands (default: 128)')
    parser.add_argument('--f_min', type=float, default=0.0,
                        help='Lowest frequency in Hz (default: 0.0)')
    parser.add_argument('--f_max', type=float, default=22050.0,
                        help='Highest frequency in Hz (default: 22050.0)')
    args = parser.parse_args()

    # Read int16 LE PCM and normalize to float32
    y = np.fromfile(args.input_pcm, dtype='<i2').astype(np.float32) / 32768.0

    # Compute mel spectrogram
    result = compute_melspectrogram(
        y,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
    )

    # Write float32 LE row-major output
    result.astype('<f4').tofile(args.output_bin)


if __name__ == '__main__':
    main()

from librosa import load, stft, istft
from scipy.io import wavfile
import numpy as np
import sys


def phasor(angles: np.ndarray, *, mag=None) -> np.ndarray:
    return mag * (np.cos(angles) + 1j * np.sin(angles)) if mag is not None else (np.cos(angles) + 1j * np.sin(angles))


def phase_vocoder(D: np.ndarray, rate: float) -> np.ndarray:
    n_fft = 2 * (D.shape[-2] - 1)
    hop_length = int(n_fft // 4)
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[-2])
    phase_acc = np.angle(D[Ellipsis, 0])
    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    d = np.pad(D, padding, mode="constant")
    for t, step in enumerate(time_steps):
        columns = d[Ellipsis, int(step): int(step + 2)]
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[Ellipsis, 0]) + alpha * np.abs(columns[Ellipsis, 1])
        d_stretch[Ellipsis, t] = phasor(phase_acc, mag=mag)
        dphase = np.angle(columns[Ellipsis, 1]) - np.angle(columns[Ellipsis, 0]) - phi_advance
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc += phi_advance + dphase

    return d_stretch


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0:
        raise ValueError("rate must positive")

    stft_data = stft(y)
    stft_stretch = phase_vocoder(
        stft_data,
        rate=rate,
    )
    len_stretch = int(round(y.shape[-1] / rate))
    y_stretch = istft(stft_stretch, dtype=y.dtype, length=len_stretch)
    return y_stretch


if __name__ == "__main__":
    input_wav = sys.argv[1]
    output_wav = sys.argv[2]
    ratio = float(sys.argv[3])
    if ratio.is_integer():
        ratio = int(ratio)
    lr_speech_data, lr_speech_rate = load(input_wav)
    stretched = time_stretch(lr_speech_data, rate=ratio)
    wavfile.write(output_wav, int(lr_speech_rate), stretched)

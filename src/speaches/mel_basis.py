from __future__ import annotations

import numpy as np


def hz2mel(hz: float) -> float:
    """Convert Hz to Mel scale."""
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel: float) -> float:
    """Convert Mel scale to Hz."""
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_mel_basis(
    sample_rate: int = 16000,
    n_fft: int = 400,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Create a Mel filter bank.
    
    This creates a matrix to convert FFT bins to Mel bins.
    """
    if fmax is None:
        fmax = float(sample_rate) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)

    # Create Mel filter bank
    fftfreqs = np.linspace(0, sample_rate / 2, int(1 + n_fft // 2))
    mel_f = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    fdiff = np.diff(mel2hz(mel_f))
    ramps = np.subtract.outer(mel2hz(mel_f), fftfreqs)

    for i in range(n_mels):
        # Lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel2hz(mel_f[2:]) - mel2hz(mel_f[:-2]))
    weights *= enorm[:, np.newaxis]

    return weights


if __name__ == "__main__":
    # Generate and save the mel basis matrix
    mel_basis = get_mel_basis()
    np.save("mel_basis.npy", mel_basis) 
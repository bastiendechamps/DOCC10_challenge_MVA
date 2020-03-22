import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import config


def audio_to_melspectrogram(audio):
    """Transform a single audio signal to a mel-pectrogram.
    Args:
        - audio : 1D np.array
    Returns:
        - spectrogram : 2D np.array
    """
    spectrogram = librosa.feature.melspectrogram(
        audio,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    return spectrogram


def audio_to_mfcc(audio):
    """Transform a single audio signal to MFCC features.
    Args:
        - audio : 1D np.array
    Returns:
        - MFCC features : 2D np.array
    """
    spectrogram = librosa.feature.mfcc(
        audio,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_mfcc=config.n_mfcc,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    spectrogram = spectrogram.astype(np.float32)

    return spectrogram


def normalize_melspectrograms(mels):
    if config.normalize_global:
        return (mels - config.mean()) / config.std()
    elif config.normalize_sample:
        axis = tuple([i + 1 for i in range(len(mels.shape) - 1)])
        return (mels - mels.mean(axis=axis, keepdims=True)) / mels.std(
            axis=axis, keepdims=True
        )
    else:
        return (mels - config.mean) / config.std
        # return (mels - mels.mean(0, keepdims=True)) / mels.std(0, keepdims=True)


def show_melspectrogram(mels):
    """Plot a spectrogram."""
    librosa.display.specshow(
        mels,
        x_axis="time",
        y_axis="mel",
        sr=config.sample_rate,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.show()

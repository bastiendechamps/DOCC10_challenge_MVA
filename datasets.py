import time
import torch
import random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset

from audio import audio_to_melspectrogram, normalize_melspectrograms
from utils import load_data
import config


N_WORKERS = mp.cpu_count()


def get_train_data():
    X, y = load_data(config.n_per_class, shuffle=False)
    X = list(X)

    # Get spectrograms from audio inputs
    with mp.Pool(N_WORKERS) as pool:
        images_list = pool.map(audio_to_melspectrogram, X)

    X = np.array(images_list)

    # Normalize spectrograms
    X = normalize_melspectrograms(X)

    return X, y


def get_test_data():
    X = list(np.load(config.test_audio_path))

    # Get spectrograms from audio inputs
    with mp.Pool(N_WORKERS) as pool:
        images_list = pool.map(audio_to_melspectrogram, X)

    X = np.array(images_list)

    # Normalize spectrograms
    X = normalize_melspectrograms(X)

    return X


class DOCC10Dataset(Dataset):
    def __init__(self, mels, y, transforms=None):
        super(DOCC10Dataset, self).__init__()
        self.mels = mels
        self.labels = y
        self.transforms = transform

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        data = self.mels[idx].astype(np.float32)
        data = np.expand_dims(data, axis=2)
        data = self.transforms(data)

        label = self.labels[idx]
        label = torch.from_numpy(label).float()

        return data, label


if __name__ == "__main__":
    X, y = get_train_data()

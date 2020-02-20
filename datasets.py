import time
import torch
import random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from audio import audio_to_melspectrogram, normalize_melspectrograms
from utils import load_data
import config


N_WORKERS = mp.cpu_count()


def get_mels_data():
    X_train, y_train, X_val, y_val = load_data(config.n_per_class, shuffle=False)
    X_train = list(X_train)
    X_val = list(X_val)

    # Get spectrograms from audio inputs
    with mp.Pool(N_WORKERS) as pool:
        images_list_train = pool.map(audio_to_melspectrogram, X_train)
        images_list_val = pool.map(audio_to_melspectrogram, X_val)

    X_train = np.array(images_list_train)
    X_val = np.array(images_list_val)

    # Normalize spectrograms
    X_train = normalize_melspectrograms(X_train)
    X_val = normalize_melspectrograms(X_val)

    return X_train, y_train, X_val, y_val


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
        self.transforms = transforms

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
    X, y, _, _ = get_mels_data()

    transforms = transforms.Compose([transforms.ToTensor()])
    dataset = DOCC10Dataset(X, y, transforms=transforms)

    print(len(dataset))
    print(dataset[0][0].size())

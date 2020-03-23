import time
import torch
import random
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset
from torchvision.transforms import transforms as trfs

from audio import audio_to_melspectrogram, normalize_melspectrograms, audio_to_mfcc
from utils import load_data, show_spectrograms, crop_center
import config


features_func = audio_to_mfcc if config.use_mfcc else audio_to_melspectrogram


def get_mels_data(shuffle=False):
    X_train, y_train, X_val, y_val = load_data(config.n_per_class, shuffle=shuffle)
    X_train = list(X_train)
    X_val = list(X_val)

    # Get spectrograms from audio inputs
    with mp.Pool(config.n_workers) as pool:
        images_list_train = pool.map(features_func, X_train)
        images_list_val = pool.map(features_func, X_val)

    X_train = np.array(images_list_train)
    X_val = np.array(images_list_val)

    # Normalize spectrograms
    X_train = normalize_melspectrograms(X_train)
    X_val = normalize_melspectrograms(X_val)

    return X_train, y_train, X_val, y_val


def get_test_data():
    X = list(np.load(config.test_audio_path))

    # Center on clicks if specified
    if config.center_on_click:
        with mp.Pool(config.n_workers) as pool:
            X = pool.map(crop_center, X)

    # Get spectrograms from audio inputs
    with mp.Pool(config.n_workers) as pool:
        X = pool.map(features_func, X)

    X = np.array(X)

    # Normalize spectrograms
    X = normalize_melspectrograms(X)

    return X


class DOCC10Dataset(Dataset):
    def __init__(self, mels, y=None, transforms=None, train=True):
        super(DOCC10Dataset, self).__init__()
        self.mels = mels
        self.labels = y
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        data = self.mels[idx].astype(np.float32)
        if self.transforms is not None:
            data = self.transforms(data)
        if self.train:
            label = self.labels[idx]
            label = torch.tensor(label).long()

            return data, label

        else:
            return data


if __name__ == "__main__":
    X, y, _, _ = get_mels_data()
    print(X.mean())
    print(X.std())
    transforms = trfs.Compose([trfs.ToTensor()])
    print(transforms(X[0]).shape)
    dataset = DOCC10Dataset(X, y, transforms=transforms)

    print(len(dataset))
    print(dataset[0][0].size())

    classes = [8, 8, 8]
    show_spectrograms(X, y, classes, 7)

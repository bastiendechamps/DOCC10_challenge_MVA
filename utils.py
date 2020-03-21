import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import multiprocessing as mp
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d


def show_spectrograms(X, y, classes, n_plots=5):
    """Plot random spectrograms from the given data."""
    fig, axes = plt.subplots(n_plots, len(classes), figsize=(10, 2 * n_plots))
    for j in range(len(classes)):
        X_sub = X[y == classes[j]]
        axes[0, j].set_title(config.classes[classes[j]])
        for i in range(n_plots):
            idx = np.random.randint(len(X_sub))
            mel = X_sub[idx]
            axes[i, j].imshow(mel)
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
    plt.show()


def load_labels_one_hot(path):
    """Load the labels in a one hot encoded np.array format."""
    df_y = pd.read_csv(path, index_col=0).values.flatten()
    labels = np.zeros((len(df_y), config.n_class))
    for i in range(len(df_y)):
        labels[i, config.class2id[df_y[i]]] = 1.0

    return labels


def load_labels(path):
    """Load the labels in a one hot encoded np.array format."""
    df_y = pd.read_csv(path, index_col=0).values.flatten()
    labels = np.array([config.class2id[df_y[i]] for i in range(len(df_y))]).astype(int)

    return labels


def load_data(n_per_class, shuffle=True):
    """Partially load train data files by keeping the class balanced"""
    X = np.load(config.train_audio_path)
    y = load_labels(config.train_labels_path)

    # Center on click

    X_train, X_val = [], []
    y_train, y_val = [], []
    for i in range(config.n_class):
        idx = np.where(y == i)
        X_i = X[idx]
        y_i = y[idx]

        # Only center the clicks that belong to 'PM' class because the others are already centered
        if config.center_on_click:
            center_func = (
                crop_center if i == config.class2id["PM"] else crop_center_middle
            )
            with mp.Pool(config.n_workers) as pool:
                X_i = pool.map(center_func, list(X_i))
            X_i = np.array(X_i)

        if shuffle:
            perm = np.random.choice(np.arange(len(X_i)), n_per_class, replace=False)
        else:
            perm = np.arange(n_per_class)

        X_i = X_i[perm, :]
        y_i = y_i[perm]

        train_idx_split = int(n_per_class * (1.0 - config.val_ratio))
        X_train.append(X_i[:train_idx_split])
        y_train.append(y_i[:train_idx_split])
        X_val.append(X_i[train_idx_split:])
        y_val.append(y_i[train_idx_split:])

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.hstack(y_train)
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.hstack(y_val)

    return X_train, y_train, X_val, y_val


def get_center(x):
    """Find the position of the click in the raw signal."""
    # Parameters
    win_size = 50
    gaussian_std = 0.5
    nyq = config.sample_rate // 2
    order = 2
    normal_cutoff = config.butter_cutoff / nyq

    # Butterworth highpass filter
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    x_h = signal.filtfilt(b, a, x)

    # Wiener filter
    x_w = signal.wiener(x_h, mysize=win_size)

    # Gaussian filter
    x_g = gaussian_filter1d(np.abs(x_w), gaussian_std)

    # Argmax of the resulting signal
    center = x_g.argmax()

    return center


def crop_center(x):
    """Crop the signal in a window aound the detected click."""
    center = get_center(x)
    window = config.center_window
    if center + window > len(x) or center - window < 0:
        center = len(x) // 2
    return x[center - window : center + window]


def crop_center_middle(x):
    """Crop the signal in a window around 4096 (without detecting the middle)."""
    return x[4096 - config.center_window : 4096 + config.center_window]


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data(10, shuffle=True)

    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)

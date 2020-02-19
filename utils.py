import pandas as pd
import numpy as np
import config


def load_labels(path):
    """Load the labels in a one hot encoded np.array format."""
    df_y = pd.read_csv(path, index_col=0).values.flatten()
    labels = np.zeros((len(df_y), config.n_class))
    for i in range(len(df_y)):
        labels[i, config.class2id[df_y[i]]] = 1.0

    return labels


def load_data(n_per_class, shuffle=True):
    """Partially load train data files by keeing the class balanced"""
    X = np.load(config.train_audio_path)
    y = load_labels(config.train_labels_path)
    X_sub = []
    y_sub = []
    for i in range(config.n_class):
        idx = y[:, i].astype(bool)
        X_i = X[idx]
        y_i = y[idx]
        if shuffle:
            perm = np.random.choice(np.arange(len(X_i)), n_per_class, replace=False)
            X_i = X_i[perm, :]

        X_sub.append(X_i[:n_per_class])
        y_sub.append(y_i[:n_per_class])

    X_sub = np.concatenate(X_sub, axis=0)
    y_sub = np.concatenate(y_sub, axis=0)

    return X_sub, y_sub


if __name__ == "__main__":
    X, y = load_data(5, shuffle=True)
    print(X.shape)
    print(y)

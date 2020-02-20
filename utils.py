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
    """Partially load train data files by keeping the class balanced"""
    X = np.load(config.train_audio_path)
    y = load_labels(config.train_labels_path)
    X_train, X_val = [], []
    y_train, y_val = [], []
    for i in range(config.n_class):
        idx = y[:, i].astype(bool)
        X_i = X[idx]
        y_i = y[idx]
        if shuffle:
            perm = np.random.choice(np.arange(len(X_i)), n_per_class, replace=False)
        else:
            perm = np.arange(n_per_class)

        X_i = X_i[perm, :]
        y_i = y_i[perm, :]

        train_idx_split = int(n_per_class * (1.0 - config.val_ratio))
        X_train.append(X_i[:train_idx_split])
        y_train.append(y_i[:train_idx_split])
        X_val.append(X_i[train_idx_split:])
        y_val.append(y_i[train_idx_split:])

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data(10, shuffle=True)
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)

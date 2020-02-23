from pathlib import Path
import torch


# File paths
train_path = Path("data/DOCC10_train/")
test_path = Path("data/DOCC10_test/")
train_audio_path = train_path / "DOCC10_Xtrain.npy"
train_labels_path = train_path / "DOCC10_Ytrain.csv"
test_audio_path = test_path / "DOCC10_Xtest.npy"
submission_path = "data/submissions/"

classes = ["UDA", "GG", "GMA", "LA", "UDB", "ZC", "ME", "SSP", "PM", "MB"]
class2id = dict(zip(classes, range(len(classes))))
n_class = len(classes)

n_per_class = 1000  # Number of samples to load, for debugging


# Audio config
sample_rate = 512000
n_mels = 40
n_fft = 512
hop_length = n_fft // 4
fmin = 1e4
fmax = 2.5e5


# Preprocessing
normalize_global = True
mean = -79.307335
std = 6.260134
val_ratio = 0.2


# Model settings
use_gpu = True
device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
model_path = "models/"

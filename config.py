from pathlib import Path
import torch
from torchvision import transforms


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

n_per_class = 3000  # Number of samples to load, for debugging : 11312 au max


# Audio config
sample_rate = 200000
n_mels = 64
n_mfcc = 20
n_fft = 256
hop_length = n_fft // 4
fmin = 1e4
fmax = 1e5
use_mfcc = False


# Preprocessing
normalize_global = False
normalize_sample = True  # not took into account if normalize_global is True
mean = -83.680084
std = 6.203564
val_ratio = 0.2

# Augmentations
# train_transform = transforms.Compose(
#     [transforms.ToPILImage(), transforms.RandomCrop(64), transforms.ToTensor()]
# )
# val_transform = transforms.Compose(
#     [transforms.ToPILImage(), transforms.CenterCrop(64), transforms.ToTensor()]
# )
train_transform = transforms.Compose([transforms.ToTensor()])
val_transform = transforms.Compose([transforms.ToTensor()])

# Model settings
use_gpu = True
device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
model_path = "models/"

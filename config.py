from pathlib import Path
import torch
from torchvision import transforms
import multiprocessing as mp


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

n_per_class = 10  # Number of samples to load, for debugging : 11312 au max

n_workers = mp.cpu_count()

# Audio config
sample_rate = 200000
center_on_click = False
butter_cutoff = 10000  # Butterworth cutoff (to find position of clicks)
center_window = 2 ** 7
n_mels = 64
n_mfcc = 20
n_fft = 256
hop_length = n_fft // 4
fmin = 1e4
fmax = 1e5
use_mfcc = False
use_scaleo = False
scaleo_width = 2 ** 8
scaleo_min_period = 1
scaleo_max_period = 20
scaleo_wavelet = "cmor1-1.5"


# Preprocessing
normalize_global = False
normalize_sample = True  # not took into account if normalize_global is True
val_ratio = 0.2
mean = -82.70759
std = 7.229028

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

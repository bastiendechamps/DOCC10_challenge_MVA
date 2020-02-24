import time
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

import config
from models import ConvModel
from datasets import get_test_data, DOCC10Dataset


def make_submission(model, model_name, sub_name, transform=None, batch_size=32):
    """Create a .csv file with the test set predictions ready to be upload."""
    # Load the model weigths
    model.load_state_dict(
        torch.load(os.path.join(config.model_path, model_name + ".pth"))
    )
    model = model.to(config.device)
    model.eval()

    # Create a dataset for the test data
    X_test = get_test_data()
    if transform is None:
        transform = transforms.ToTensor()
    test_dataset = DOCC10Dataset(X_test, transforms=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []

    # Predict classes
    for X in test_loader:
        X = X.to(config.device)
        y_pred = model(X).view(-1, config.n_class).detach()
        _, pred_classes = torch.max(y_pred, dim=1)
        pred_classes = pred_classes.cpu().numpy()
        all_preds.append(pred_classes)

    all_preds = np.hstack(all_preds)
    pred_classes = [config.classes[idx] for idx in all_preds]

    # Make the submission
    sub = pd.DataFrame(
        {"ID": list(range(len(pred_classes))), "TARGET": pred_classes},
        columns=["ID", "TARGET"],
    )
    sub.to_csv(os.path.join(config.submission_path, sub_name + ".csv"), index=False)


if __name__ == "__main__":
    model_name = "larger"
    sub_name = "all_data_larger"

    # Initialize the model
    model = ConvModel(config.n_class)

    # make submission
    make_submission(model, model_name, sub_name)

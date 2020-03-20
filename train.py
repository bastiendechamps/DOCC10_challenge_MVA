import time
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from datasets import DOCC10Dataset, get_mels_data
from models import ConvModel


def train(
    model,
    optimizer,
    criterion,
    train_dataset,
    val_dataset,
    nepoch=20,
    batch_size=32,
    print_every=1,
    model_name="model",
):
    """Train the model."""
    model.to(config.device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    start = time.time()

    best_acc = 0.0

    for epoch in range(nepoch):
        # Train phase
        model.train()
        avg_train_loss = 0.0
        avg_train_acc = 0.0

        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(config.device), y.to(config.device)
            y_pred = model(X).view(-1, config.n_class)
            loss = criterion(y_pred.float(), y.long())
            loss.backward()
            avg_train_loss += loss.item() / len(train_dataset)
            optimizer.step()

            # Compute accuracy
            _, pred_classes = torch.max(y_pred.cpu(), dim=1)
            acc = (pred_classes == y.cpu()).sum().item()
            avg_train_acc += acc / len(train_dataset)

        # Val phase
        model.eval()
        avg_val_loss = 0.0
        avg_val_acc = 0.0

        for X, y in val_loader:
            X, y = X.to(config.device), y.to(config.device)
            with torch.no_grad():
                y_pred = model(X).view(-1, config.n_class)
                loss = criterion(y_pred.float(), y.long())
                avg_val_loss += loss.item() / len(val_dataset)

                # Compute validation accuracy
                _, pred_classes = torch.max(y_pred, dim=1)
                val_acc = (pred_classes.cpu() == y.cpu()).sum().item()
                avg_val_acc += val_acc / len(val_dataset)

        if (epoch + 1) % print_every == 0:
            elapsed_time = time.time() - start
            start = time.time()
            print(
                f"epoch [{epoch + 1}/{nepoch}]   loss {avg_train_loss:.4f}   acc {avg_train_acc:.4f}   val loss {avg_val_loss:.4f}   val acc {avg_val_acc:.4f}   time {elapsed_time:.1f}s"
            )

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(
                model.state_dict(), os.path.join(config.model_path, model_name + ".pth")
            )


def eval_model(model, dataset, batch_size=32):
    """Evaluate a model on a dataset. Return the accuracy and the probabilities."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    avg_acc = 0.0

    for X, y in loader:
        X = X.to(config.device)
        with torch.no_grad():
            y_pred = model(X).view(-1, config.n_class)
        all_preds.append(y_pred.cpu().numpy())

        # Compute validation accuracy
        _, pred_classes = torch.max(y_pred, dim=1)
        acc = (pred_classes.cpu() == y).sum().item()
        avg_acc += acc / len(dataset)

    return avg_acc, np.concatenate(all_preds, axis=0)


if __name__ == "__main__":
    # Data
    print("Building spectrograms...")
    transform = transforms.Compose([transforms.ToTensor()])
    start = time.time()
    X_train, y_train, X_val, y_val = get_mels_data()
    print("done in {:.2f}s".format(time.time() - start))
    train_dataset = DOCC10Dataset(X_train, y_train, transforms=transform)
    val_dataset = DOCC10Dataset(X_val, y_val, transforms=transform)

    # Model
    model = ConvModel(num_classes=config.n_class)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    # Training
    model_name = "test"
    print("Training model...")
    train(
        model,
        optimizer,
        criterion,
        train_dataset,
        val_dataset,
        10,
        batch_size=32,
        model_name=model_name,
    )

    # model.load_state_dict(torch.load("models/test.pth"))
    # model.to(config.device)
    # acc, _ = eval_model(model, val_dataset)
    # print("Validation accuracy:", acc)

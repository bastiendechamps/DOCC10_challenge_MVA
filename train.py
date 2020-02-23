import time
import os
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
    nepoch=50,
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

        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(config.device), y.to(config.device)
            y_pred = model(X).view(-1, config.n_class)
            loss = criterion(y_pred.float(), y.long())
            loss.backward()
            avg_train_loss += loss.item() / len(train_dataset)
            optimizer.step()

        # Val phase
        model.eval()
        avg_val_loss = 0.0
        avg_val_acc = 0.0

        for X, y in val_loader:
            X, y = X.to(config.device), y.to(config.device)
            y_pred = model(X).view(-1, config.n_class).detach()
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
                f"epoch [{epoch + 1}/{nepoch}]   loss {avg_train_loss:.4f}   val loss {avg_val_loss:.4f}   val acc {avg_val_acc:.4f}   time {elapsed_time:.1f}s"
            )

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(
                model.state_dict(), os.path.join(config.model_path, model_name + ".pth")
            )


if __name__ == "__main__":
    # Data
    transforms = transforms.Compose([transforms.ToTensor()])
    X_train, y_train, X_val, y_val = get_mels_data()
    train_dataset = DOCC10Dataset(X_train, y_train, transforms=transforms)
    val_dataset = DOCC10Dataset(X_val, y_val, transforms=transforms)

    # Model
    model = ConvModel(num_classes=config.n_class)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    train(model, optimizer, criterion, train_dataset, val_dataset, 10)

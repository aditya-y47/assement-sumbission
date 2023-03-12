import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.manual_seed(121)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(121)

batch_size = 32


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CONFIG_NAME = input(">Enter Model Config Name: ")


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = torch.tensor(self.data.iloc[index, :].values, dtype=torch.float32)
        y = torch.tensor(self.targets.iloc[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data)


pth = "./data/master_data_deberta.csv"
master_data = pd.read_csv(pth)
master_data.drop("review", axis=1, inplace=True)
targets = master_data.pop("targets")
X_train, X_test, y_train, y_test = train_test_split(
    master_data, targets, test_size=0.3, stratify=targets
)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.667, stratify=y_test)
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


class Network(nn.Module):
    def __init__(self, input_feats: int = 770) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_feats, 2**10)
        self.fc2 = nn.Linear(2**10, 2**8)
        self.fc3 = nn.Linear(2**8, 2**5)
        self.fc4 = nn.Linear(2**5, 2**3)
        self.fc5 = nn.Linear(2**3, 2**3)
        self.final = nn.Linear(2**3, 1)
        self.droup_out = nn.Dropout(0.41170711473010735)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.droup_out(x)
        x = F.selu(self.fc3(x))
        x = self.droup_out(x)
        x = F.selu(self.fc4(x))
        x = F.selu(self.fc5(x))
        x = self.final(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.sigmoid(logits)
            pred = prob > 0.5
        return pred


def train(model, optimizer, criterion, scheduler, train_dl, device):
    model.train()
    total_loss, total_correct = 0.0, 0.0
    with tqdm(train_dl, desc="Training", unit="batch") as tepoch:
        for data, target in tepoch:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            total_correct += torch.sum(
                torch.round(torch.sigmoid(output.squeeze())) == target.byte()
            )
            tepoch.set_postfix(
                loss=total_loss / len(train_dl.dataset),
                acc=total_correct / len(train_dl.dataset),
            )
        scheduler.step(total_loss / len(train_dl.dataset))
        training_loss = total_loss / len(train_dl.dataset)
        training_accuracy = total_correct / len(train_dl.dataset)
        logging.info(
            f"Training loss: {training_loss:.4f}, training accuracy: {training_accuracy:.4f}"
        )
        return training_loss, training_accuracy


def validate(model, criterion, val_dl, device):
    model.eval()
    val_loss = 0.0
    total_predictions = []
    total_targets = []
    with torch.no_grad():
        for inputs, labels in val_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            val_loss += loss.item()
            predictions = (outputs > 0).int()
            total_predictions.extend(predictions.cpu().numpy().tolist())
            total_targets.extend(labels.cpu().numpy().tolist())

        val_accuracy = accuracy_score(total_targets, total_predictions)
        val_precision = precision_score(total_targets, total_predictions)
        val_recall = recall_score(total_targets, total_predictions)
        val_f1 = f1_score(total_targets, total_predictions)

    # Log the validation results to a JSON file
    results = {
        "val_loss": val_loss / len(val_dl),
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
    }
    with open(f"./results/{CONFIG_NAME}.json", "w") as f:
        json.dump(results, f, indent=4)

    return val_loss / len(val_dl), val_accuracy, val_precision, val_recall, val_f1


def train_loop(
    model,
    optimizer,
    criterion,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    num_epochs,
):
    # initialize lists for metrics
    training_losses = []
    training_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        training_loss, training_accuracy = train(
            model, optimizer, criterion, scheduler, train_dataloader, device
        )
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)

        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(
            model, criterion, val_dataloader, device=device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./models/best_model_{CONFIG_NAME}.pt")

        logging.info(
            f"Epoch {epoch+1} - Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}, Validation precision: {val_precision:.4f}, Validation recall: {val_recall:.4f}, Validation F1: {val_f1:.4f}"
        )

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.plot(training_losses, label="Training Loss", color="blue")
    plt.legend()
    plt.title("Training Loss")
    plt.xlabel("Epochs")

    plt.subplot(2, 3, 2)
    plt.plot(val_losses, label="Validation Loss", color="green")
    plt.legend()
    plt.title("Validation Loss")
    plt.xlabel("Epochs")

    plt.subplot(2, 3, 3)
    plt.plot(val_accuracies, label="Validation Accuracy", color="red")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")

    plt.subplot(2, 3, 4)
    plt.plot(val_precisions, label="Validation Precision", color="purple")
    plt.legend()
    plt.title("Validation Precision")
    plt.xlabel("Epochs")

    plt.subplot(2, 3, 5)
    plt.plot(val_recalls, label="Validation Recall", color="orange")
    plt.legend()
    plt.title("Validation Recall")
    plt.xlabel("Epochs")

    plt.subplot(2, 3, 6)
    plt.plot(val_f1s, label="Validation F1 Score", color="brown")
    plt.legend()
    plt.title("Validation F1 Score")
    plt.xlabel("Epochs")

    plt.tight_layout()
    plt.savefig(f"./metrics_{CONFIG_NAME}.png")


if __name__ == "__main__":
    # pass
    device = "cpu"
    model = Network(input_feats=1026)
    optim = torch.optim.Adam(model.parameters(), lr=0.00019937232653442498)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        patience=3,
        verbose=True,
    )
    train_loop(
        model=model,
        optimizer=optim,
        criterion=criterion,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=24,
    )

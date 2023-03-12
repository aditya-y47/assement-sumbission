import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from train import Network, test_dataloader
import json

CONFIG_NAME = input(">Enter Model Config Name: ")


def inference(model, criterion, val_dl, device):
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
            predictions = model.predict(inputs)
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
    with open(f"./results/test_time_pref_{CONFIG_NAME}.json", "w") as f:
        json.dump(results, f, indent=4)

    return None


model = Network(input_feats=1026)
model.load_state_dict(torch.load("./models/best_model_hparam_tuned_debeta.pt"))
# optim = torch.optim.AdamW(model.parameters(), lr=1e-3)S
criterion = nn.BCEWithLogitsLoss()
model.eval()
metrics = inference(model=model, criterion=criterion, val_dl=test_dataloader, device="cpu")

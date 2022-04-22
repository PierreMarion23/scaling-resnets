import pytorch_lightning as pl
import torch


def get_prediction(data, model: pl.LightningModule, device):
    model.eval() # Deactivates gradient graph construction during eval.
    data = data.to(device)
    model.to(device)
    probabilities = torch.softmax(model(data), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities


def get_true_targets_predictions(test_dl, model, device):
    true_targets, predictions = [], []
    for batch in iter(test_dl):
        data, target = batch
        true_targets.extend(target)
        prediction, _ = get_prediction(data, model, device)
        predictions.extend(prediction.cpu())
    return true_targets, predictions


def get_eval_loss(test_dl, model, device):
    model.eval()
    loss = 0
    for n, batch in enumerate(test_dl):
        data, target = batch
        logits = model(data.to(device))
        loss += model.loss(logits, target.to(device))
    return loss / n
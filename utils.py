import pytorch_lightning as pl
import torch


def get_prediction(data, model: pl.LightningModule):
    model.eval() # Deactivates gradient graph construction during eval.
    probabilities = torch.softmax(model(data), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities

def get_true_targets_predictions(test_dl, model):
    true_targets, predictions = [], []
    for batch in iter(test_dl):
        data, target = batch
        true_targets.extend(target)
        prediction, _ = get_prediction(data, model)
        predictions.extend(prediction.cpu())
    return true_targets, predictions

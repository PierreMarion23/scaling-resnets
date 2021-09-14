import pytorch_lightning as pl
import sklearn.metrics
import torch
from tqdm.autonotebook import tqdm


def get_prediction(data, model: pl.LightningModule):
    model.freeze() # prepares model for predicting
    probabilities = torch.softmax(model(data), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities

def print_classification_report(test_dl, model):
    true_targets, predictions = [], []
    for batch in tqdm(iter(test_dl), total=len(test_dl)):
        data, target = batch
        true_targets.extend(target)
        prediction, _ = get_prediction(data, model)
        predictions.extend(prediction.cpu())
    print(sklearn.metrics.classification_report(true_targets, predictions, digits=3))

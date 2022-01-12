import copy

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class ResNet(pl.LightningModule):
    def __init__(self, initial_width, final_width, **model_config):
        super().__init__()

        self.initial_width = initial_width
        self.final_width = final_width
        self._model_config = model_config
        self.width = model_config['width']
        self.depth = model_config['depth']
        self.activation = getattr(nn, model_config['activation'])()  # e.g. torch.nn.ReLU()
        self.train_init = model_config['train_init']
        self.train_final = model_config['train_final']

        self.init = nn.Linear(self.initial_width, self.width)
        if not self.train_init:
            self.init.weight.requires_grad = False
            self.init.bias.requires_grad = False
        self.layers = nn.Sequential(
            *[nn.Linear(self.width, self.width) for _ in range(self.depth)])
        self.final = nn.Linear(self.width, self.final_width)
        if not self.train_final:
            self.final.weight.requires_grad = False
            self.final.bias.requires_grad = False

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        hidden_state = self.init(x)
        for k in range(self.depth):
            hidden_state = hidden_state + self.layers[k](self.activation(hidden_state)) / self.depth
        return self.final(hidden_state)

    def training_step(self, batch, batch_no):
        self.train()
        data, target = batch
        logits = self(data)
        loss = self.loss(logits, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=0.005)

    def copy(self):
        result = ResNet(self.initial_width, self.final_width, **self._model_config)
        result.load_state_dict(copy.deepcopy(self.state_dict()))
        return result


def polyfit(series, degree):
    time = np.arange(len(series))
    polynomial = np.poly1d(np.polyfit(time, series, degree))
    poly_approx = polynomial(time)
    error = np.linalg.norm(series - poly_approx)
    return poly_approx, error


# TODO: vectorize the for loops.
def denoise_weights_bias(layers, degree):
    depth = len(layers)
    width = layers[0].weight.shape[0]
    errors_weights = np.zeros((width, width))
    errors_bias = np.zeros((width))
    denoised_weights = np.zeros((depth, width, width))
    denoised_bias = np.zeros((depth, width))

    for i in range(width):
        bias = np.array([layers[k].bias[i].detach().numpy() for k in range(depth)])
        polyfit_bias, error_bias = polyfit(bias, degree)
        denoised_bias[:, i] = polyfit_bias
        errors_bias[i] = error_bias

        for j in range(width):
            weight = np.array([layers[k].weight[i, j].detach().numpy() for k in range(depth)])
            polyfit_weight, error_weight = polyfit(weight, degree)
            denoised_weights[:, i, j] = polyfit_weight
            errors_weights[i, j] = error_weight

    return denoised_weights, denoised_bias, errors_weights, errors_bias

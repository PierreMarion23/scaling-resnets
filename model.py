import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class ResNet(pl.LightningModule):
    def __init__(self, initial_width,
                 width, depth, final_depth, activation, train_init=True, train_final=True):
        super().__init__()
        self.init = nn.Linear(initial_width, width)
        if not train_init:
            self.init.weight.requires_grad = False
            self.init.bias.requires_grad = False
        self.layers = nn.Sequential(
            *[nn.Linear(width, width) for _ in range(depth)])
        self.depth = depth
        self.activation = activation
        self.final = nn.Linear(width, final_depth)
        if not train_final:
            self.final.weight.requires_grad = False
            self.final.bias.requires_grad = False
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        hidden_state = self.init(x.reshape(-1, 28*28))
        for i in range(self.depth):
            hidden_state = hidden_state + self.layers[i](self.activation(hidden_state)) / self.depth
        return self.final(hidden_state)

    def training_step(self, batch, batch_no):
        data, target = batch
        logits = self(data)
        loss = self.loss(logits, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=0.005)


def polyfit(series, degree):
    time = np.arange(len(series))
    polynomial = np.poly1d(np.polyfit(time, series, degree))
    poly_approx = polynomial(time)
    error = np.linalg.norm(series - poly_approx)
    return poly_approx, error


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

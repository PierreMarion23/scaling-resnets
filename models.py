import copy
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch import Tensor


def create_linear_layer(in_features, out_features, noise_mult=1, bias=True):
    # Linear layers are initalized with a symmetric uniform distribution, hence the noise level can be scaled as follows.
    layer = nn.Linear(in_features, out_features, bias=bias)
    length = torch.sqrt(torch.Tensor([3. / in_features])) * noise_mult
    layer.weight = nn.Parameter(2 * length * torch.rand((out_features, in_features)) - length)
    if bias:
        layer.bias = nn.Parameter(2 * length * torch.rand((out_features,)) - length)
    return layer


# TODO: create a parent class for common methods between both models.
class FCResNet(pl.LightningModule):
    def __init__(self, first_coord, final_width, **model_config):
        super().__init__()

        self.initial_width = first_coord
        self.final_width = final_width
        self._model_config = model_config
        self.width = model_config['width']
        self.depth = model_config['depth']
        self.scaling = model_config['scaling']
        self.activation = getattr(nn, model_config['activation'])()  # e.g. torch.nn.ReLU()
        self.train_init = model_config['train_init']
        self.train_final = model_config['train_final']

        # For ReZero, add trainable scaling parameters on each layer
        if self.scaling == "rezero":
            self.scaling_weight = nn.Parameter(torch.zeros(self.depth), requires_grad=True)
        elif self.scaling == 'beta':
            self.scaling_weight = torch.full((self.depth,), 1 / (float(self.depth) ** model_config['scaling_beta']))
        else:
            self.scaling_weight = torch.full((self.depth,), 1)


        # Uniform initialization on [-sqrt(3/width), sqrt(3/width)]
        self.init = create_linear_layer(self.initial_width, self.width, model_config['init_final_initialization_noise'])
        self.layers = nn.Sequential(
            *[create_linear_layer(self.width, self.width, model_config['layers_initialization_noise'])
              for _ in range(self.depth)])
        self.outer_weights = nn.Sequential(
            *[create_linear_layer(self.width, self.width, model_config['layers_initialization_noise'], bias=False)
              for _ in range(self.depth)])
        self.final = create_linear_layer(self.width, self.final_width, model_config['init_final_initialization_noise'])

        if not self.train_init:
            self.init.weight.requires_grad = False
            self.init.bias.requires_grad = False
        if not self.train_final:
            self.final.weight.requires_grad = False
            self.final.bias.requires_grad = False
        if self._model_config['batch_norm']:
            self.batch_norms = nn.Sequential(
                *[nn.BatchNorm1d(self.width) for _ in range(self.depth)])
        # ReZero initialization
        # torch.nn.init.kaiming_normal_(self.init.weight, a=0, mode='fan_in', nonlinearity='relu')
        # for i in range(self.depth):
        #         torch.nn.init.xavier_normal_(self.layers[i].weight, gain=torch.sqrt(torch.tensor(2.)))

        self.loss = nn.CrossEntropyLoss()

    def forward_hidden_state(self, hidden_state):
        # Function that outputs the last hidden state, useful to compare norms
        for k in range(self.depth):
            if self._model_config['batch_norm']:
                normed_hidden_state = self.batch_norms[k](hidden_state)
            else:
                normed_hidden_state = hidden_state
            if self._model_config['skip_connection']:
                hidden_state = hidden_state + self.scaling_weight[k] * self.outer_weights[k](
                    self.activation(self.layers[k](normed_hidden_state)))
            else:
                hidden_state = self.scaling_weight[k] * self.outer_weights[k](
                    self.activation(self.layers[k](normed_hidden_state)))
        return hidden_state

    def forward(self, x):
        hidden_state = self.init(x)
        hidden_state = self.forward_hidden_state(hidden_state)
        # hidden_state = self.final(hidden_state)
        # return hidden_state / torch.norm(hidden_state)
        return self.final(hidden_state)

    def training_step(self, batch, batch_no):
        self.train()
        data, target = batch
        logits = self(data)
        loss = self.loss(logits, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adagrad(
        #     filter(lambda p: p.requires_grad, self.parameters()), lr=self._model_config['lr'])
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1.0)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self._model_config['lr'])
        return optimizer
        # return torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=0.01)
        #

    def copy(self):
        result = FCResNet(self.initial_width, self.final_width, **self._model_config)
        result.load_state_dict(copy.deepcopy(self.state_dict()))
        return result


def conv3x3(in_planes: int, out_planes: int) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=1,
        bias=False,
    )


class BasicBlock(pl.LightningModule):
    def __init__(
        self,
        channels: int,
        norm_layer: Callable[..., nn.Module],
        depth: int
    ) -> None:
        super().__init__()
        self.bn1 = norm_layer(channels)
        self.conv1 = conv3x3(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.depth = depth

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = x + out / self.depth
        return out


class ConvResNet(pl.LightningModule):
    def __init__(self, first_coord, final_width, **model_config) -> None:
        super().__init__()
        self._norm_layer = nn.BatchNorm2d

        self.channels = model_config['channels']
        self.depth = model_config['depth']

        self.conv1 = nn.Conv2d(first_coord, self.channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential(*[BasicBlock(self.channels, self._norm_layer, self.depth) for _ in range(self.depth)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels, final_width)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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
        result = FCResNet(self.initial_width, self.final_width, **self._model_config)
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


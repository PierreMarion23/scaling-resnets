from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

import utils


def create_linear_layers_rbf(
        depth: int, width: int, regularity: float) -> nn.Sequential:
    """Initialize the weights of a sequence of layers of fixed width as
    discretizations of a smooth Gaussian process with rbf kernel.

    :param depth: depth of the ResNet
    :param width: width of the layers
    :param regularity: variance of the rbf kernel
    :return: initialized layers of the ResNet as a nn.Sequential object
    """
    mean = [0] * (depth + 1)
    gram = utils.gram_matrix(depth, regularity)
    gp = np.random.default_rng().multivariate_normal(
        mean, gram, (width, width))
    increments = gp[:, :, 1:] - gp[:, :, :-1]
    weights = torch.Tensor(
        increments / (
                np.mean(np.std(increments, axis=(0, 1))) * np.sqrt(width)
        )
    )
    layers = [
        nn.Linear(width, width, bias=False) for _ in range(depth)]
    for k in range(depth):
        layers[k].weight = torch.nn.Parameter(weights[:,:,k])
    return nn.Sequential(*layers)


def create_linear_layers_fbm(
        depth: int, width: int, hurst_index: float) -> nn.Sequential:
    """Initialize the weights of a sequence of layers of fixed width as
    increments of a fractional Brownian motion.

    :param depth: depth of the ResNet
    :param width: width of the layers
    :param hurst_index: Hurst index of the fractional Brownian motion
    :return: initialized layers of the ResNet as a nn.Sequential object
    """
    weights = torch.zeros(depth, width, width)
    for i in range(width):
        for j in range(width):
            weights[:, i, j] = torch.Tensor(
                utils.generate_fbm(depth, hurst_index)[1] / np.sqrt(
                    width))
    layers = [nn.Linear(width, width) for _ in range(depth)]
    for k in range(depth):
        layers[k].weight = torch.nn.Parameter(weights[k])
        layers[k].bias = torch.nn.Parameter(torch.zeros(width,))
    return nn.Sequential(*layers)


def create_linear_layer(
        in_features: int, out_features:int, bias: bool = True) -> nn.Linear:
    """Initialize one linear layer with a symmetric uniform distribution on
    [-sqrt(3/in_features), sqrt(3/in_features)]

    :param in_features: size of the input to the layer
    :param out_features: size of the output of the layer
    :param bias: whether to include a bias
    :return:
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    length = torch.sqrt(torch.Tensor([3. / in_features]))
    layer.weight = nn.Parameter(
        2 * length * torch.rand((out_features, in_features)) - length)
    if bias:
        layer.bias = nn.Parameter(
            2 * length * torch.rand((out_features,)) - length)
    return layer


class ResNet(pl.LightningModule, ABC):
    def __init__(self, first_coord: int, final_width:int,
                 **model_config: dict):
        """General class of residual neural network

        :param first_coord: size of the input data
        :param final_width: size of the output
        :param model_config: configuration dictionary with hyperparameters
        """
        super().__init__()

        self.initial_width = first_coord
        self.final_width = final_width
        self.model_config = model_config
        self.width = model_config['width']
        self.depth = model_config['depth']
        self.activation = getattr(nn, model_config['activation'])()

        self.scaling_weight = torch.full(
            (self.depth,), 1 / (
                    float(self.depth) ** model_config['scaling_beta']))

        # Uniform initialization on [-sqrt(3/width), sqrt(3/width)]
        self.init = create_linear_layer(
            self.initial_width, self.width, bias=False)
        self.final = create_linear_layer(
            self.width, self.final_width, bias=False)

        self.loss = nn.CrossEntropyLoss()

    def reset_scaling(self, beta: float):
        """ Reset the scaling parameter as 1/depth ** beta

        :param beta: new value for the scaling parameter beta
        :return:
        """
        self.scaling_weight = torch.full((self.depth,),
                                         1 / (float(self.depth) ** beta))

    @abstractmethod
    def forward_hidden_state(self, hidden_state):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = self.init(x)
        hidden_state = self.forward_hidden_state(hidden_state)
        return self.final(hidden_state)

    def training_step(self, batch, batch_no):
        self.train()
        data, target = batch
        logits = self(data)
        loss = self.loss(logits, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.model_config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.model_config['step_lr'])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}


class SimpleResNet(ResNet):
    def __init__(
            self, first_coord: int, final_width:int, **model_config: dict):
        """Residual neural network, subclass of ResNet where the weights can
        be initialized as discretizations of stochastic processes and the
        update function consists of a non-linearity and one matrix
        multiplication (called 'res-1' in the paper).

        :param first_coord: size of the input data
        :param final_width: size of the output
        :param model_config: configuration dictionary with hyperparameters
        """
        super().__init__(first_coord, final_width, **model_config)

        if model_config['regularity']['type'] == 'iid':
            self.inner_weights = nn.Sequential(
                *[create_linear_layer(self.width, self.width, bias=False)
                  for _ in range(self.depth)])
            self.outer_weights = nn.Sequential(
                *[create_linear_layer(self.width, self.width, bias=False)
                  for _ in range(self.depth)])
        elif model_config['regularity']['type'] == 'fbm':
            self.outer_weights = create_linear_layers_fbm(
                self.depth, self.width, model_config['regularity']['value'])
        elif model_config['regularity']['type'] == 'rbf':
            self.outer_weights = create_linear_layers_rbf(
                self.depth, self.width, model_config['regularity']['value'])
        else:
            raise ValueError(
                "argument regularity['type'] should be one of 'iid', 'fbm', "
                "'rbf'")

    def forward_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Function that outputs the last hidden state, useful to compare norms

        :param hidden_state: output of the initial layer
        :return: output of the last hidden layer
        """
        for k in range(self.depth):
            hidden_state = hidden_state + self.scaling_weight[k] * \
                           self.outer_weights[k](self.activation(hidden_state))
        return hidden_state


class FullResNet(ResNet):
    def __init__(
            self, first_coord: int, final_width: int, **model_config: dict):
        """Residual neural network, subclass of ResNet where the weights can
        be initialized as discretizations of stochastic processes and where we
        add a matrix multiplication in the update function, compared to
        SimpleResNet (called 'res-3' in the paper).

        :param first_coord: size of the input data
        :param final_width: size of the output
        :param model_config: configuration dictionary with hyperparameters
        """
        super().__init__(first_coord, final_width, **model_config)

        if model_config['regularity']['type'] == 'iid':
            self.inner_weights = nn.Sequential(
                *[create_linear_layer(self.width, self.width, bias=False)
                  for _ in range(self.depth)])
            self.outer_weights = nn.Sequential(
                *[create_linear_layer(self.width, self.width, bias=False)
                  for _ in range(self.depth)])
        elif model_config['regularity']['type'] == 'fbm':
            self.inner_weights = create_linear_layers_fbm(
                self.depth, self.width, model_config['regularity']['value'])
            self.outer_weights = create_linear_layers_fbm(
                self.depth, self.width, model_config['regularity']['value'])
        elif model_config['regularity']['type'] == 'rbf':
            self.inner_weights = create_linear_layers_rbf(
                self.depth, self.width, model_config['regularity']['value'])
            self.outer_weights = create_linear_layers_rbf(
                self.depth, self.width, model_config['regularity']['value'])
        else:
            raise ValueError(
                "argument regularity['type'] should be one of 'iid', 'fbm', "
                "'rbf'")

        self.final = create_linear_layer(
            self.width, self.final_width, bias=False)

        self.loss = nn.CrossEntropyLoss()

    def forward_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Function that outputs the last hidden state, useful to compare norms

        :param hidden_state: output of the initial layer
        :return: output of the last hidden layer
        """
        for k in range(self.depth):
            hidden_state = hidden_state + self.scaling_weight[k] * self.outer_weights[k](
                    self.activation(self.inner_weights[k](hidden_state)))
        return hidden_state

from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import utils
from typing import Optional

def create_linear_layers_rbf(
        depth: int, width: int, bandwidth: float,
        seed: Optional[int] = None) -> nn.Sequential:
    """Initialize the weights of a sequence of layers of fixed width as
    discretizations of a smooth Gaussian process with rbf kernel.

    :param depth: depth of the ResNet
    :param width: width of the layers
    :param regularity: variance of the rbf kernel
    :return: initialized layers of the ResNet as a nn.Sequential object
    """
    mean = [0] * (depth + 1)
    cov_matrix = utils.cov_matrix_for_rbf_kernel(depth, bandwidth)
    weights = np.random.default_rng(seed = seed).multivariate_normal(
        mean, cov_matrix, (width, width)) / np.sqrt(width)
    layers = [
        nn.Linear(width, width, bias=False) for _ in range(depth)]
    for k in range(depth):
        layers[k].weight = torch.nn.Parameter(torch.Tensor(weights[:, :, k]))
    return nn.Sequential(*layers)


def create_linear_layers_rbf_with_cov(
        depth: int, width: int, bandwidth: float, cov: np.array) -> nn.Sequential:
    """Initialize the weights of a sequence of layers of fixed width as
    discretizations of a smooth Gaussian process with rbf kernel.

    :param depth: depth of the ResNet
    :param width: width of the layers
    :param regularity: variance of the rbf kernel
    :return: initialized layers of the ResNet as a nn.Sequential object
    """
    mean = [0] * (depth + 1)
    cov_matrix = utils.cov_matrix_for_rbf_kernel(depth, bandwidth)
    weights = np.random.default_rng().multivariate_normal(
        mean, cov_matrix, (width, width)) / np.sqrt(width)
    layers = [
        nn.Linear(width, width, bias=False) for _ in range(depth)]
    for k in range(depth):
        layers[k].weight = torch.nn.Parameter(
            torch.matmul(
                torch.linalg.cholesky(torch.Tensor(cov)),
                torch.Tensor(weights[:, :, k]).reshape(width*width)
            ).reshape(width, width)
        )
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

def create_linear_layer_volterra(
        depth: int, width: int, hurst: float, init: float,
        T: float = 1.0, lam: float = 0.3, mu: float=0.3,
        theta: float = 0.02) -> nn.Sequential:
    """Initialize the weights of a sequence of layers of fixed width
    as rough votality model.

    :param depth: depth of the ResNet
    :prama width: width of the layers
    :param hurst: hurst index of the rough votality model
    :param n: the nth step of the simulation
    :return: initialized layers of the ResNet as a nn.Sequential object
    """
    weights = torch.zeros(depth, width, width)
    process = utils.generate_heston_paths(1, hurst, steps = depth, v_0=init)
    for i in range(width):
        for j in range(width):
            weights[:, i, j] = torch.Tensor((process-init)/np.sqrt(width))[1:]
    print("layer created")

    layers = [nn.Linear(width, width) for _ in range(depth)]
    for k in range(depth):
        layers[k].weight = torch.nn.Parameter(weights[k])
        layers[k].bias = torch.nn.Parameter(torch.zeros(width, ))
    return nn.Sequential(*layers)
    

def create_linear_layer(
        in_features: int, out_features:int, bias: bool = True,
        in_distro: bool = False, scaling_factor: float = 0.0) -> nn.Linear:
    """Initialize one linear layer with a normal distribution of
    variance 1 / in_features

    :param in_features: size of the input to the layer
    :param out_features: size of the output of the layer
    :param bias: whether to include a bias
    :param in_distro: whether to put scaling factor in the distribution of the weight
    :param scaling_factor: rescale the std of the distro of the weights only used if
                           in_distro is set to be True
    :return:
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    if in_distro:
        layer.weight = nn.Parameter(
            torch.normal(mean = torch.zeros(out_features, in_features),
                         std = torch.full(size = (out_features, in_features),
                                          fill_value=scaling_factor / np.square(in_features))
                         )
        )
    else:
        layer.weight = nn.Parameter(
            torch.randn(out_features, in_features) / np.sqrt(in_features))
    if bias:
        layer.bias = nn.Parameter(
            torch.randn(out_features,) / np.sqrt(in_features)) 
    return layer

def create_zero_layer(
        in_features: int, out_features: int):
    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight = nn.Parameter(torch.zeros(out_features, in_features))
    return layer

def create_linear_layer_with_cov(
        in_features: int, out_features: int, cov: np.array, depth: int,
        bias: bool = True) -> nn.Sequential:
    """Initialize one linear layer with a normal distribution of
    variance 1 / in_features with covariance

    :param in_features: size of the input to the layer
    :param out_features: size of the output of the layer
    :param depth: depth of the NN
    :param bias: whether to include a bias
    :param cov: covariance matrix for normal weight
    :return
    """
    layers = [nn.Linear(in_features, out_features, bias=bias) for _ in range(depth)]
    sample = np.random.multivariate_normal(
        np.zeros(shape=in_features*out_features), cov, size=depth)
    # print(sample[0, :])
    for k in range(depth):
        layers[k].weight = nn.Parameter(
            torch.Tensor(sample[k, :].reshape((out_features, in_features))).to(torch.float)
        )
        if bias:
            layers[k].bias = nn.Parameter(
                torch.randn(out_features,) / np.sqrt(in_features))
    return nn.Sequential(*layers)


    
    
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

        if model_config['activation'] == 'LeakyReLU':
            self.activation = nn.LeakyReLU(
                negative_slope=model_config['negative_slope'])
        else:
            self.activation = getattr(nn, model_config['activation'])()

        self.scaling_weight = torch.full(
            (self.depth,), 1 / (
                    float(self.depth) ** model_config['scaling']))

        # Uniform initialization on [-sqrt(3/width), sqrt(3/width)]
        self.init = create_linear_layer(
            self.initial_width, self.width, bias=False)
        self.final = create_linear_layer(
            self.width, self.final_width, bias=False)
        if model_config.get('loss'):
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def reset_scaling(self, scaling: float):
        """ Reset the scaling parameter as 1/depth ** scaling

        :param scaling: new value for the scaling parameter
        :return:
        """
        self.scaling_weight = torch.full((self.depth,),
                                         1 / (float(self.depth) ** scaling))

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

class LinearResNet(ResNet):
    def __init__(
            self, first_coord: int, final_width:int, seed: Optional[int] = None,
            half: bool = False, **model_config: dict):
        """Residual neural network, subclass of ResNet where the weights can
        be initialized as iid Gaussian variables and with linear layer.
        Providing a seed will produce same layer weights for both halved and whole
        layers.

        :param first_coord: size of the input data
        :param final_width: size of the output
        :param model_config: configuration dictionary with hyperparameters
        """
        if not seed is None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super().__init__(first_coord, final_width, **model_config)
        
        self.half = half
        if model_config['regularity']['type'] == 'iid':
            weights = nn.Sequential(
                *[create_linear_layer(self.width, self.width, bias=False)
                for _ in range(self.depth)])

        
        elif model_config['regularity']['type'] == 'zero':
            weights = nn.Sequential(
                *[create_zero_layer(self.width, self.width)
                for _ in range(self.depth)]
            )
        elif model_config['regularity']['type'] == 'smooth':
            weights = create_linear_layers_rbf(
                        self.depth, self.width,
                        model_config['regularity']['value'],
                        seed = seed
                    )
        else:
            raise ValueError(
                "Argument regularity['type'] should be one of 'iid', 'zero' or 'smooth'.")
        
        self.weights = weights if not half \
            else nn.Sequential(
            *[weights[k] if k%2 == 0 else None for k in range(self.depth)])
        d = 2 if half else 1
        self.scaling_weight = torch.full(
            (self.depth,), (d / (float(self.depth)) ** model_config['scaling']))
    
    def forward_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Function that outputs the last hidden state, useful to compare norms

        :param hidden_state: output of the initial layer
        :return: output of the last hidden layer
        """
        for k in range(self.depth):
            if self.weights[k]:
                hidden_state = hidden_state + self.scaling_weight[k] * \
                   self.weights[k](hidden_state)
        return hidden_state

class SimpleResNet(ResNet):
    def __init__(
            self, first_coord: int, final_width:int, seed: Optional[int] = None,
            half: Optional[bool] = None, **model_config: dict):
        """Residual neural network, subclass of ResNet where the weights can
        be initialized as discretizations of stochastic processes and the
        update function consists of a non-linearity and one matrix
        multiplication (called 'res-1' in the paper).

        :param first_coord: size of the input data
        :param final_width: size of the output
        :param model_config: configuration dictionary with hyperparameters
        """
        if not seed is None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super().__init__(first_coord, final_width, **model_config)

        if model_config['regularity']['type'] == 'iid':
            self.outer_weights = nn.Sequential(
                *[create_linear_layer(self.width, self.width, bias=False)
                  for _ in range(self.depth)])
        elif model_config['regularity']['type'] == 'fbm':
            self.outer_weights = create_linear_layers_fbm(
                self.depth, self.width, model_config['regularity']['value'])
        elif model_config['regularity']['type'] == 'rbf':
            self.outer_weights = create_linear_layers_rbf(
                self.depth, self.width, model_config['regularity']['value'],
                seed)
        elif model_config['regularity']['type'] == 'volterra':
            self.outer_weights = create_linear_layer_volterra(
                self.depth, self.width, 
                hurst=model_config['regularity']['hurst'],
                init = model_config['regularity']['init'],
            )
        else:
            raise ValueError(
                "argument regularity['type'] should be one of 'iid', 'fbm', "
                "'rbf'")
        self.outer_weights = self.outer_weights if not half \
            else nn.Sequential(
            *[self.outer_weights[k] if k%2 == 0 else None for k in range(self.depth)])
        d = 2 if half else 1
        self.scaling_weight = torch.full(
            (self.depth,), (d / (float(self.depth)) ** model_config['scaling']))

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
            self, first_coord: int, final_width: int, seed: Optional[int] = None,
            half: Optional[bool] = None, **model_config: dict):
        """Residual neural network, subclass of ResNet where the weights can
        be initialized as discretizations of stochastic processes and where we
        add a matrix multiplication in the update function, compared to
        SimpleResNet (called 'res-3' in the paper).

        :param first_coord: size of the input data
        :param final_width: size of the output
        :param model_config: configuration dictionary with hyperparameters
        """
        if not seed is None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super().__init__(first_coord, final_width, **model_config)

        self.half = half
        self.in_distro = True if model_config.get('in_distro') else False
        self.cov = model_config.get('cov')
        if model_config['regularity']['type'] == 'iid':
            self.inner_weights = nn.Sequential(
                *[create_linear_layer(self.width, self.width, bias=False)
                  for _ in range(self.depth)])
            self.outer_weights = nn.Sequential(
                *[create_linear_layer(self.width, self.width, bias=False,
                                      in_distro = self.in_distro,
                                      scaling_factor = self.scaling_weight[k])
                  for k in range(self.depth)])
        elif model_config['regularity']['type'] == 'iid_with_corr':
            self.inner_weights = create_linear_layer_with_cov(
                self.width, self.width, cov=model_config['cov'],
                depth=self.depth, bias=False
            )
            self.outer_weights = create_linear_layer_with_cov(
                self.width, self.width, cov=model_config['cov'],
                depth=self.depth, bias=False
            )
        elif model_config['regularity']['type'] == 'fbm':
            self.inner_weights = create_linear_layers_fbm(
                self.depth, self.width, model_config['regularity']['value'])
            self.outer_weights = create_linear_layers_fbm(
                self.depth, self.width, model_config['regularity']['value'])
        elif model_config['regularity']['type'] == 'rbf':
            self.inner_weights = create_linear_layers_rbf(
                self.depth, self.width, model_config['regularity']['value'],
                seed=seed)
            self.outer_weights = create_linear_layers_rbf(
                self.depth, self.width, model_config['regularity']['value'],
                seed=seed)
        elif model_config['regularity']['type'] == 'rbf_with_corr':
            self.inner_weights = create_linear_layers_rbf_with_cov(
                self.depth, self.width, model_config['regularity']['value'],
                cov=model_config['cov']
            )
            self.outer_weights = create_linear_layers_rbf_with_cov(
                self.depth, self.width, model_config['regularity']['value'],
                cov=model_config['cov']
            )
        elif model_config['regularity']['type'] == 'volterra':
            self.inner_weights = create_linear_layer_volterra(
                self.depth, self.width, 
                hurst = model_config['regularity']['hurst'],
                init = model_config['regularity']['init']
            )
            self.outer_weights = create_linear_layer_volterra(
                self.depth, self.width, 
                hurst = model_config['regularity']['hurst'],
                init = model_config['regularity']['init']
            )
        else:
            raise ValueError(
                "argument regularity['type'] should be one of 'iid', 'fbm', "
                "'rbf', 'volterra'")
        if half:
            self.inner_weights = nn.Sequential(
                *[self.inner_weights[k] if k%2==0 else None
                  for k in range(self.depth)]
            )
            self.outer_weights = nn.Sequential(
                *[self.outer_weights[k] if k%2==0 else None
                  for k in range(self.depth)]
            )

        d = 2 if half else 1
        self.scaling_weight = torch.full(
            (self.depth,), (d / (float(self.depth)) ** model_config['scaling']))

        self.final = create_linear_layer(
            self.width, self.final_width, bias=False)

        #self.loss = nn.CrossEntropyLoss()

    def forward_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Function that outputs the last hidden state, useful to compare norms

        :param hidden_state: output of the initial layer
        :return: output of the last hidden layer
        """
        if self.in_distro:
            for k in range(self.depth):
                if not self.outer_weights[k] is None:
                    hidden_state = hidden_state + (
                        self.outer_weights[k](self.activation(self.inner_weights[k](hidden_state)))
                    )
        else:
            for k in range(self.depth):
                if not self.outer_weights[k] is None:
                    hidden_state = hidden_state + (
                            self.scaling_weight[k] *
                            self.outer_weights[k](
                                self.activation(self.inner_weights[k](hidden_state))
                            )
                    )
        return hidden_state

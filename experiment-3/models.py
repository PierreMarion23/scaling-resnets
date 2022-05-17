import copy
from statistics import mode
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


def rbf_kernel(x1, x2, variance):
    return np.exp(-1 * ((x1-x2) ** 2) / (2*variance))

def gram_matrix(depth ,variance):
    xs = np.linspace(0, 1, depth + 1)
    return [[rbf_kernel(x1,x2,variance) for x2 in xs] for x1 in xs]


def create_linear_layer_specific_regularity_smooth(depth, in_features, out_features, regularity=0.01):
    weights = torch.zeros(depth, in_features, out_features)
    mean = [0]*(depth +1)
    gram = gram_matrix(depth, regularity)
    gp = np.random.default_rng().multivariate_normal(mean, gram, (in_features, out_features))
    weights = torch.Tensor(gp / np.sqrt(in_features))
    layers = [nn.Linear(in_features, out_features, bias=False) for _ in range(depth)]
    for k in range(depth):
        layers[k].weight = torch.nn.Parameter(weights[:,:,k])
        layers[k].bias = torch.nn.Parameter(torch.zeros(out_features,))
    return nn.Sequential(*layers)


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
        self.smooth = model_config['smooth'] if 'smooth' in model_config else False

        # For ReZero, add trainable scaling parameters on each layer
        if self.scaling == "rezero":
            self.scaling_weight = nn.Parameter(torch.zeros(self.depth), requires_grad=True)
        elif self.scaling == 'beta':
            self.scaling_weight = torch.full((self.depth,), 1 / (float(self.depth) ** model_config['scaling_beta']))
        else:
            self.scaling_weight = torch.full((self.depth,), 1)


        # Uniform initialization on [-sqrt(3/width), sqrt(3/width)]
        self.init = create_linear_layer(self.initial_width, self.width, model_config['init_final_initialization_noise'])
        if not self.smooth:
            self.layers = nn.Sequential(
                *[create_linear_layer(self.width, self.width, model_config['layers_initialization_noise'])
                for _ in range(self.depth)])
            self.inner_weights = nn.Sequential(
            *[create_linear_layer(self.width, self.width, model_config['layers_initialization_noise'])
              for _ in range(self.depth)])
        else:
            self.layers = create_linear_layer_specific_regularity_smooth(self.depth, self.width, self.width, model_config['layers_initialization_noise'])
            self.inner_weights = create_linear_layer_specific_regularity_smooth(self.depth, self.width, self.width, model_config['layers_initialization_noise'])
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
                hidden_state = hidden_state + self.scaling_weight[k] * self.layers[k](
                    self.activation(self.inner_weights[k](normed_hidden_state)))
             #   hidden_state = hidden_state + self.scaling_weight[k] * self.layers[k](self.activation(normed_hidden_state))
            else:
                hidden_state = self.scaling_weight[k] * self.layers[k](
                    self.activation(self.inner_weights[k](normed_hidden_state)))
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

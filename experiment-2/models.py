import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


def davies_harte(N, H):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method

    args:
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k, H: 0.5 * (
                np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(
            k + 1) ** (2 * H))
    g = [gamma(k, H) for k in range(0, N)];
    r = g + [0] + g[::-1][0:N - 1]

    # Step 1 (eigenvalues)
    j = np.arange(0, 2 * N);
    k = 2 * N - 1
    lk = np.fft.fft(
        r * np.exp(2 * np.pi * complex(0, 1) * k * j * (1 / (2 * N))))[::-1]

    # Step 2 (get random variables)
    Vj = np.zeros((2 * N, 2), dtype=complex);
    Vj[0, 0] = np.random.standard_normal();
    Vj[N, 0] = np.random.standard_normal()

    for i in range(1, N):
        Vj1 = np.random.standard_normal();
        Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1;
        Vj[i][1] = Vj2;
        Vj[2 * N - i][0] = Vj1;
        Vj[2 * N - i][1] = Vj2

    # Step 3 (compute Z)
    wk = np.zeros(2 * N, dtype=complex)
    wk[0] = np.sqrt((lk[0] / (2 * N))) * Vj[0][0];
    wk[1:N] = np.sqrt(lk[1:N] / (4 * N)) * (
                (Vj[1:N].T[0]) + (complex(0, 1) * Vj[1:N].T[1]))
    wk[N] = np.sqrt((lk[0] / (2 * N))) * Vj[N][0]
    wk[N + 1:2 * N] = np.sqrt(lk[N + 1:2 * N] / (4 * N)) * (
                np.flip(Vj[1:N].T[0]) - (
                    complex(0, 1) * np.flip(Vj[1:N].T[1])))

    Z = np.fft.fft(wk);
    fGn = Z[0:N]
    fBm = np.cumsum(fGn) * (N ** (-H))
    path = np.array([0] + list(fBm))
    return path, fGn


def create_linear_layer_specific_regularity(depth, in_features, out_features, regularity):
    weights = torch.zeros(depth, in_features, out_features)
    for i in range(in_features):
        for j in range(out_features):
            weights[:, i, j] = torch.Tensor(davies_harte(depth, regularity)[1] / np.sqrt(in_features))
    layers = [nn.Linear(in_features, out_features) for _ in range(depth)]
    for k in range(depth):
        layers[k].weight = torch.nn.Parameter(weights[k])
        layers[k].bias = torch.nn.Parameter(torch.zeros(out_features,))
    return nn.Sequential(*layers)


def create_linear_layer(in_features, out_features, bias=True):
    # Linear layers are initalized with a symmetric uniform distribution, hence the noise level can be scaled as follows.
    layer = nn.Linear(in_features, out_features, bias=bias)
    length = torch.sqrt(torch.Tensor([3. / in_features]))
    layer.weight = nn.Parameter(2 * length * torch.rand((out_features, in_features)) - length)
    if bias:
        layer.bias = nn.Parameter(2 * length * torch.rand((out_features,)) - length)
    return layer


class FCResNet(pl.LightningModule):
    def __init__(self, first_coord, final_width, **model_config):
        super().__init__()

        self.initial_width = first_coord
        self.final_width = final_width
        self.model_config = model_config
        self.width = model_config['width']
        self.depth = model_config['depth']
        self.activation = getattr(nn, model_config[
            'activation'])()  # e.g. torch.nn.ReLU()
        self.regularity = model_config['regularity']

        self.scaling_weight = torch.full((self.depth,), 1 / (
                    float(self.depth) ** model_config['scaling_beta']))

        # Uniform initialization on [-sqrt(3/width), sqrt(3/width)]
        self.init = create_linear_layer(self.initial_width, self.width,
                                        bias=False)
        self.outer_weights = create_linear_layer_specific_regularity(
                self.depth, self.width, self.width, self.regularity)
        self.final = create_linear_layer(self.width, self.final_width,
                                         bias=False)

        self.loss = nn.CrossEntropyLoss()

    def reset_scaling(self, beta):
        self.scaling_weight = torch.full((self.depth,),
                                         1 / (float(self.depth) ** beta))

    def forward_hidden_state(self, hidden_state):
        # Function that outputs the last hidden state, useful to compare norms
        for k in range(self.depth):
            hidden_state = hidden_state + self.scaling_weight[k] * \
                           self.outer_weights[k](
                               self.activation(hidden_state))
        return hidden_state

    def forward(self, x):
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

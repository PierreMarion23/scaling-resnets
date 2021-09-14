import copy
from matplotlib import pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

import data
import model
import utils



width = 30
depth = 200
resnet = model.ResNet(28*28, width, depth, 10, torch.nn.ReLU())
train_dl, test_dl = data.mnist()

if torch.cuda.is_available():
    gpu = 1
else:
    gpu = 0
trainer = pl.Trainer(
    gpus=gpu,
    max_epochs=1,
    progress_bar_refresh_rate=20
)

trainer.fit(resnet, train_dl)
utils.print_classification_report(test_dl, resnet)

layers = list(resnet.layers.children())
weight_example = np.array([layers[k].weight[8, 19].detach().numpy() for k in range(depth)])
poly_example, _ = model.polyfit(weight_example, 3)
plt.plot(weight_example)
plt.plot(poly_example)
plt.show()
plt.plot(weight_example - poly_example)
plt.show()

denoised_weights, denoised_bias, errors_weights, errors_bias = model.denoise_weights_bias(layers, 3)
print(np.mean(errors_weights))
print(np.mean(errors_bias))
plt.hist(errors_weights.flatten(), density=True)
plt.hist(errors_bias.flatten(), density=True, alpha=0.5)
plt.show()

denoised_model = copy.deepcopy(resnet)

new_weights = list(denoised_model.layers.children())
for k in range(depth):
    new_weights[k].weight = torch.nn.Parameter(torch.zeros((width, width)))
    new_weights[k].bias = torch.nn.Parameter(torch.zeros((width,)))

utils.print_classification_report(test_dl, denoised_model)

new_weights = list(denoised_model.layers.children())
for k in range(depth):
    new_weights[k].weight = torch.nn.Parameter(torch.Tensor(denoised_weights[k]))
    new_weights[k].bias = torch.nn.Parameter(torch.Tensor(denoised_bias[k]))

utils.print_classification_report(test_dl, denoised_model)

import os

import pytorch_lightning as pl
import torch
import wandb

import logs
import data
import model

# os.environ['WANDB_MODE'] = 'offline'
wandb.login()
wandb.init(project='scaling-resnets', entity='lpsm-deep')

# datasets = {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
train_dl, test_dl, data_size, nb_classes = data.load_dataset('MNIST')

width = 30
depth = 200
resnet = model.ResNet(data_size, width, depth, nb_classes, torch.nn.ReLU(), test_dl=test_dl)

if torch.cuda.is_available():
    gpu = 1
else:
    gpu = 0

wandb_logger = pl.loggers.WandbLogger()
trainer = pl.Trainer(
    gpus=gpu,
    max_epochs=30,
    progress_bar_refresh_rate=20,
    logger=wandb_logger,
    callbacks=[logs.PrintingCallback()],
    checkpoint_callback=False
)

trainer.fit(resnet, train_dl)

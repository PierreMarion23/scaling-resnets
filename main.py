import os

import pytorch_lightning as pl
import torch
import wandb

import config
import logs
import data
import model

# os.environ['WANDB_MODE'] = 'offline'
wandb.login()
wandb.init(project='scaling-resnets', entity='lpsm-deep')

train_dl, test_dl, data_size, nb_classes = data.load_dataset(config.standard_config['dataset'])

resnet = model.ResNet(initial_width=data_size, final_width=nb_classes, **config.standard_config['model'], test_dl=test_dl)

gpu = 1 if torch.cuda.is_available() else 0

wandb_logger = pl.loggers.WandbLogger()
trainer = pl.Trainer(
    gpus=gpu,
    max_epochs=config.standard_config['epochs'],
    progress_bar_refresh_rate=20,
    logger=wandb_logger,
    callbacks=[logs.PrintingCallback()],
    checkpoint_callback=False
)

trainer.fit(resnet, train_dl)

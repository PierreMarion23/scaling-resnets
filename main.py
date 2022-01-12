import os

import click
import pytorch_lightning as pl
import torch
import wandb

import config
import logs
import data
import model


@click.command()
@click.option('--config', '-c', 'config_name', default='standard', prompt='Name of the config to use',
              help='Name of the config to use.')
@click.option('--offline', '-o', is_flag=True, help='Do not sync the wandb run online.')
def fit(config_name, offline):
    if offline:
        os.environ['WANDB_MODE'] = 'offline'
    wandb.login()
    wandb.init(project='scaling-resnets', entity='lpsm-deep')

    config_dict = getattr(config, config_name)

    train_dl, test_dl, data_size, nb_classes = data.load_dataset(config_dict['dataset'])

    resnet = model.ResNet(initial_width=data_size, final_width=nb_classes, **config_dict['model'])

    gpu = 1 if torch.cuda.is_available() else 0

    wandb_logger = pl.loggers.WandbLogger()
    trainer = pl.Trainer(
        gpus=gpu,
        max_epochs=config_dict['epochs'],
        progress_bar_refresh_rate=20,
        logger=wandb_logger,
        callbacks=[logs.PrintingCallback(test_dl)],
        checkpoint_callback=False
    )
    trainer.fit(resnet, train_dl)


if __name__ == '__main__':
    fit()
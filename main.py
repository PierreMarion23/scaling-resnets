import os

import click
import pytorch_lightning as pl
import torch
import wandb

import config
import logs
import data
import models


@click.command()
@click.option('--config', '-c', 'config_name', default='standard', prompt='Name of the config to use',
              help='Name of the config to use.')
@click.option('--offline', '-o', is_flag=True, help='Do not sync the wandb run online.')
def fit(config_name, offline):
    if offline:
        os.environ['WANDB_MODE'] = 'offline'
    wandb.login()

    config_dict = getattr(config, config_name)
    name_template = config_dict['name']

    for dataset in config_dict['dataset']:
        name = name_template.replace('dataset', dataset)
        with wandb.init(project='scaling-resnets', entity='lpsm-deep', name=name) as run:
            wandb.config.update(config_dict)

            train_dl, test_dl, first_coord, nb_classes = data.load_dataset(
                dataset, vectorize=(config_dict['model'] == 'FCResNet'))

            model_class = getattr(models, config_dict['model'])
            model = model_class(first_coord=first_coord, final_width=nb_classes, **config_dict['model-config'])

            gpu = 1 if torch.cuda.is_available() else 0
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            full_logs = 'full-logs' not in config_dict or config_dict['full-logs']
            wandb_logger = pl.loggers.WandbLogger()
            trainer = pl.Trainer(
                gpus=gpu,
                max_epochs=config_dict['epochs'],
                logger=wandb_logger,
                callbacks=[logs.PrintingCallback(test_dl, device, full_logs), pl.callbacks.progress.TQDMProgressBar(20)],
                enable_checkpointing=False
            )
            trainer.fit(model, train_dl)

            model_artifact = wandb.Artifact('resnet', type='model', metadata=config_dict)
            with model_artifact.new_file("trained-model.ckpt", mode="wb") as file:
                torch.save(model, file)
            run.log_artifact(model_artifact)


if __name__ == '__main__':
    fit()

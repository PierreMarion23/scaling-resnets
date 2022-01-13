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

    config_dict = getattr(config, config_name)
    name_template = config_dict['name']

    for dataset in config_dict['dataset']:
        name = name_template.replace('dataset', dataset)
        with wandb.init(project='scaling-resnets', entity='lpsm-deep', name=name) as run:
            wandb.config = config_dict

            train_dl, test_dl, data_size, nb_classes = data.load_dataset(dataset)

            resnet = model.ResNet(initial_width=data_size, final_width=nb_classes, **config_dict['model'])

            gpu = 1 if torch.cuda.is_available() else 0
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            wandb_logger = pl.loggers.WandbLogger()
            trainer = pl.Trainer(
                gpus=gpu,
                max_epochs=config_dict['epochs'],
                progress_bar_refresh_rate=20,
                logger=wandb_logger,
                callbacks=[logs.PrintingCallback(test_dl, device)],
                checkpoint_callback=False
            )
            trainer.fit(resnet, train_dl)

            model_artifact = wandb.Artifact('resnet', type='model', metadata=config_dict)
            model_artifact.aliases.append(name)
            with model_artifact.new_file("trained-model.ckpt", mode="wb") as file:
                torch.save(resnet, file)
            run.log_artifact(model_artifact)


if __name__ == '__main__':
    fit()
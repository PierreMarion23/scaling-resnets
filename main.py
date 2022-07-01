import copy
from multiprocessing import Pool
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import time
import torch

import config
import data
import models
import utils


def fit(config_dict: dict, verbose: bool = False):
    """Train a ResNet following the configuration.

    :param config_dict: configuration of the network and dataset
    :param verbose: print information about traning
    :return:
    """
    name_template = config_dict['name']

    for dataset in config_dict['dataset']:
        name = name_template.replace('dataset', dataset)

        train_dl, test_dl, first_coord, nb_classes = data.load_dataset(
            dataset, vectorize=True)

        model_class = getattr(models, config_dict['model'])
        model = model_class(
            first_coord=first_coord, final_width=nb_classes,
            **config_dict['model-config'])

        gpu = 1 if torch.cuda.is_available() else 0
        device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")

        trainer = pl.Trainer(
                gpus=gpu,
                max_epochs=config_dict['epochs'],
                enable_checkpointing=False,
                enable_progress_bar=verbose,
                enable_model_summary=verbose
            )
        trainer.fit(model, train_dl)

        print('Training finished')
        true_targets, predictions = utils.get_true_targets_predictions(
            test_dl, model, device)
        accuracy = np.mean(np.array(true_targets) == np.array(predictions))
        loss = utils.get_eval_loss(test_dl, model, device)

        metrics = {'test_accuracy': accuracy, 'test_loss': loss}
        if verbose:
            print(f'Test accuracy: {accuracy}')
        results_dir = f'{os.getcwd()}/results/{name}/{str(time.time())}'
        os.makedirs(results_dir, exist_ok=True)
        trainer.save_checkpoint(f'{results_dir}/model.ckpt')

        with open(f'{results_dir}/metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(f'{results_dir}/config.pkl', 'wb') as f:
            pickle.dump(config_dict, f)
    return model


def fit_parallel(config: dict):
    """Train in parallel ResNet with different learning rate, scaling, and
    initialization.

    :param config: configuration of the network and dataset
    :return:
    """
    grid_lr = [0.0001, 0.001, 0.01, 0.1, 1.]
    grid_regularity = np.linspace(0.1, 0.9, 10)
    grid_beta = np.linspace(0.1, 0.9, 10)

    list_configs = []
    for lr in grid_lr:
        for reg in grid_regularity:
            for beta in grid_beta:
                config['model-config']['lr'] = lr
                config['model-config']['regularity']['value'] = reg
                config['model-config']['scaling_beta'] = beta
                list_configs.append(copy.deepcopy(config))
    with Pool(processes=config['n_workers']) as pool:
        pool.map(fit, list_configs)


if __name__ == '__main__':
    fit_parallel(config.perf_weights_regularity)

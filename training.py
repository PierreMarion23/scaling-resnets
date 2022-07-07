import copy
import glob
from multiprocessing import Pool
import os
import pickle
import time
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch

import data
import models
import utils


def get_results(exp_name: str) -> list:
    """Read the results saved after execution of the training file.

    :param exp_name: name of the configuration
    :return: list of results
    """
    results = {'accuracy': [], 'regularity': [], 'lr': [], 'scaling': []}
    for directory in glob.glob(os.path.join('results', exp_name, '*')):
        with open(os.path.join(directory, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
            results['regularity'].append(
                config['model-config']['regularity']['value'])
            results['lr'].append(config['model-config']['lr'])
            results['scaling'].append(config['model-config']['scaling_beta'])
        with open(os.path.join(directory, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)
            results['accuracy'].append(metrics['test_accuracy'])

    return results


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


def fit_parallel(exp_config: dict,
                 grid_lr: list,
                 grid_regularity: list,
                 grid_beta: list,
                 resume_experiment: Optional[bool] = False):
    """Train in parallel ResNet with different learning rate, scaling, and
    initialization.

    :param config: configuration of the network and dataset
    :param grid_lr: grid of learning rates
    :param grid_regularity: grid of initialization regularities
    :param grid_beta: grid of scaling betas
    :param resume_experiment: if True, will look in the results folder if
    the grid was partially explored and skip the experiments which were
    already performed
    :return:
    """
    if resume_experiment:
        previous_results = get_results(exp_config['name'].replace('dataset', 'MNIST'))
        found_experiments = [(previous_results['lr'][k], 
                             previous_results['regularity'][k],
                             previous_results['scaling'][k]) 
                             for k in range(len(previous_results['lr']))
                            ]
    list_configs = []
    for lr in grid_lr:
        for reg in grid_regularity:
            for beta in grid_beta:
                if (lr, reg, beta) not in found_experiments:
                    exp_config['model-config']['lr'] = lr
                    exp_config['model-config']['regularity']['value'] = reg
                    exp_config['model-config']['scaling_beta'] = beta
                    list_configs.append(copy.deepcopy(exp_config))
    with Pool(processes=exp_config['n_workers']) as pool:
        pool.map(fit, list_configs)

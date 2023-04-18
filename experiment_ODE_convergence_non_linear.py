import distutils.spawn

import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import copy

from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

from torch.multiprocessing import Pool, set_start_method

from typing import Optional, Final, List, Dict
from models import FullResNet
import config
from training import multiple_fit
from utils import max_norm, mean_norm
import time
import pickle

NUM_DATA: Final[int] = 1
NUM_TEST: Final[int] = 200
WIDTH: Final[int] = 40
SEED: Final[int] = 42

#sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

def gen_data(n: int, d: int) -> Dataset:
    """generate random data

    :param n: amount of data generated
    :param d: dimension of the data
    """
    return torch.utils.data.TensorDataset(torch.randn((n, d)), torch.randn((n, 1)))

def compute_Nvs2N_diff(
        config: dict, depth: int, niter: int, activation: str,
        scaling: float, initialization_type: str, seed: int, 
        train_data, test_data, verbose: bool = False) -> dict:
    """Initialize, fit and compute their weight difference after training
    halved and whole models given the depth.

    :param config: configuration of the model
    :param depth: depth of the model
    :param niter: the number of iteration
    :param seed: the seed of the randomness
    :param train_data: the training data
    :param test_data: the test data
    :param verbose: enable verbosity

    :return: dictionary containing the result
    """
    model_config = copy.deepcopy(config['model-config'])
    model_config['depth'] = depth
    model_config['activation'] = activation
    model_config['regularity']['type'] = initialization_type
    model_config['scaling'] = scaling
    print(f"depth: {depth}, activation: {activation}, scaling: {scaling}, "
          f"init_type: {initialization_type}, round: {niter}")
    models = []
    for half in [True, False]:
        model = FullResNet(
        first_coord=config['dim_input'], final_width=config['nb_classes'],
        half = half, seed = seed, **model_config)
        model_copy = copy.deepcopy(model)
        modelN_outer_weights_init = [
            model_copy.outer_weights[2*k].weight for k in range(depth//2)]
        models.append(model)
    
    if verbose: print(f"round {niter} start")
    multiple_fit(models, seed, config['epochs'],
                                   train_data, test_data,
                                    verbose=verbose)
    model2N = models[1]
    modelN = models[0]
    if verbose: 
        print(model2N)
        print(modelN)
    model2N_outer_weights = [
        model2N.outer_weights[2*k].weight for k in range(depth//2)]
    modelN_outer_weights = [
        modelN.outer_weights[2*k].weight for k in range(depth//2)]
    model2N_inner_weights = [
        model2N.inner_weights[2*k].weight for k in range(depth//2)]
    modelN_inner_weights = [
        modelN.inner_weights[2*k].weight for k in range(depth//2)]
    if verbose:
        print(model2N_outer_weights)
        print(modelN_outer_weights)
    diffNvs2N_max = max(
        max_norm(model2N_outer_weights, modelN_outer_weights),
        max_norm(model2N_inner_weights, modelN_inner_weights)
        )
    diffNvs2N_mean = np.sqrt((
        mean_norm(model2N_outer_weights, modelN_outer_weights) + \
        mean_norm(model2N_inner_weights, modelN_inner_weights)) / 2)
    diffN_max = max_norm(modelN_outer_weights, modelN_outer_weights_init)
    diffN_mean = np.sqrt(
        mean_norm(modelN_outer_weights, modelN_outer_weights_init))
    return {'depth': depth, 'activation': activation, 'scaling': scaling,
            'type': initialization_type,
            'weight_diff_max': diffNvs2N_max,
            'weight_diff_mean': diffNvs2N_mean,
            'weight_diff_init_after_max':diffN_max,
            'weight_diff_init_after_mean': diffN_mean}


def decomp(c):
    return compute_Nvs2N_diff(*c)

def plot_results(results: List[Dict], scaling_order: List[float], 
                 activ_order: List[str], init_order: List[str],
                 exp_name: Optional[str] = None,
                 filepath: Optional[str] = 'figures'):
    df = pd.DataFrame(results)
    df.columns = ['depth', 'activ', 'scaling',
                  'type', 'weight_diff_max', 'weight_diff_mean',
                  'init_diff_max', 'init_diff_mean']

    #df_log = df.apply(np.log)
    for t in init_order:
        for norm in ['weight_diff_max', 
                     'weight_diff_mean',
                     'init_diff_max', 
                     'init_diff_mean']:
            g = sns.relplot(
                x='depth',
                y=norm,
                row = 'scaling',
                row_order = scaling_order,
                col = 'activ',
                col_order = activ_order,
                data=df[df['type']==t],
                kind='line',
                facet_kws=dict(sharey=False)
            )
            for ax, col in zip(g.axes[0], activ_order):
                ax.set_title(col)
            for ax in g.axes[1]:
                ax.set_title('')
                ax.set_xlabel(r'$L$')
            g.axes[0, 0].set_ylabel("scaling = 0.5")
            g.axes[1, 0].set_ylabel("scaling = 1.0")
            plt.savefig(
                os.path.join(filepath, f'{norm}_non_linear_type_{t}_{exp_name}.pdf'), bbox_inches='tight')
            print(f'{norm}_non_linear_type_{t}_{exp_name}.pdf saved!')

def run_test(config: dict, depth_list: list, scaling_list: list,
             activ_list: List[str], init_list: List[str],
             num_test: int = NUM_TEST, num_train: int = NUM_DATA,
             verbose: bool = False) -> list:
    """Perform the experiment on ODE convergence parallely

    :param config: configuration dictionary of the model
    :param depth_list: list of depth
    :param num_test: Number of test data
    :param num_train: Number of train data
    :param verbose: enable verbosity
    
    :return: list of containing the information of depth and weight difference
    """
    results = []
    res_list = [[(copy.deepcopy(config), 
                depth, n, activation, scaling,
                init_type,  np.random.randint(99999), 
                gen_data(num_train, config['dim_input']), 
                gen_data(num_test, config['dim_input']),
                verbose)
                for n in range(config['niter'])]
                for depth in depth_list
                for activation in activ_list
                for init_type in init_list
                for scaling in scaling_list
                ]
    for item in res_list:
        res = list(map(decomp, item))
        results.extend(res)
                    # with Pool(processes=config['n_workers']) as pool:
                    #     res = pool.map(decomp, res_list)
                    #     res_all.extend(res)
    # pool = Pool(config['n_workers'])
    # res = pool.map(decomp, res_list)
    # pool.close()
    # pool.join()
    
    return results
        

    
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    pl.seed_everything(SEED)

    model_config = config.ODE_convergence_exp_non_linear
    scaling_list = [0.5, 1.0]
    activ_list = ['ReLU', 'Sigmoid', 'Tanh']
    init_list = ['iid', 'rbf']  
    depth_list = np.linspace(10, 1000, num=10, dtype=int)
    #depth_list =  np.linspace(100, 2000, num=20, dtype = int)
    #depth_list = [20, 40, 80]
    #depth_list = [20]


    pickle_filepath = "pickles/ODE_convergence/non_linear"
    figures_filepath = "figures/ODE_convergence/non_linear"
    exp_name = "10-100030iter"
    pickle_name = f"diff_{exp_name}.pkl"
    os.makedirs(pickle_filepath, exist_ok=True)
    os.makedirs(figures_filepath, exist_ok=True)
    resume = False
    if not resume:
        print("Experiment starts")
        results_ODE = run_test(
            model_config, depth_list, 
            scaling_list, activ_list, init_list,
            verbose = False)
        print("Results Get!")

        print("Start dumping the results...")
        with open(f"{pickle_filepath}/{pickle_name}", "wb") as f:
            pickle.dump(results_ODE, f)

        print("Begin plotting...")
        plot_results(results_ODE, scaling_list, activ_list, 
                     init_list, exp_name, figures_filepath)

    else:
        print("Resuming previous experiment...")
        with open(f"{pickle_filepath}/{pickle_name}", 'rb') as f:
            results = pickle.load(f)
        print("Start plotting...")
        plot_results(results, scaling_list, activ_list,
                     init_list, exp_name, figures_filepath)
    #plot_results(results_ODE, filepath)

    # #print(results_ODE)
    print("Experiment done!")

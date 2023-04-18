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
from models import LinearResNet
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
        config: Dict, depth: int, niter: int, 
        scaling: float, init: str, seed: int, 
        train_data: Dataset, test_data: Dataset, 
        verbose: bool = False) -> Dict:
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
    model_config = config['model-config']
    model_config['depth'] = depth
    model_config['scaling'] = scaling
    model_config['regularity']['type'] = init
    print(f"depth: {depth}, scaling: {scaling}, type: {init}")
    models = []
    for half in [True, False]:
        model = LinearResNet(
        first_coord=config['dim_input'], final_width=config['nb_classes'],
        half = half, seed = seed, **model_config)
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
    model2N_weights = [
        model2N.weights[2*k].weight for k in range(depth//2)]
    modelN_weights = [modelN.weights[2*k].weight for k in range(depth//2)]
    if verbose:
        print(model2N_weights)
        print(modelN_weights)
    diffNvs2N = max_norm(model2N_weights, modelN_weights)
    diffNvs2N_mean = mean_norm(model2N_weights, modelN_weights)

    return {'depth': depth, 'scaling': scaling, 'init': init,
            'weight_diff': diffNvs2N, 
            'weight_diff_mean': diffNvs2N_mean}

def compute_Nvs0_diff(
        config: dict, depth: int, niter: int, scaling: float,
        seed: int, train_data, test_data, verbose: bool = False) -> Dict:
    config['model-config']['depth'] = depth
    config['model-config']['scaling'] = scaling
    print(depth)
    if verbose: print(f"Round {niter} started")
    modelN = LinearResNet(first_coord=config['dim_input'], 
                          final_width=config['nb_classes'], 
                          seed = seed, **config['model-config'])
    config_zero = copy.deepcopy(config)
    config_zero['model-config']['regularity']['type'] = 'zero'
    model_zero = LinearResNet(first_coord=config['dim_input'], 
                            final_width=config['nb_classes'], seed = seed, 
                            **config_zero['model-config'])
    models = [modelN, model_zero]
    modelN_trained, model0_trained = multiple_fit(models, SEED, 
                                                  config['epochs'],
                                                  train_data,
                                                  test_data, verbose)

    modelN_weights = [modelN_trained.weights[k].weight for k in range(depth)]
    model0_weights = [model0_trained.weights[k].weight for k in range(depth)]
    diff = max_norm(modelN_weights, model0_weights)
    return {'depth': depth, 'weight_diff': diff}


def decomp(c):
    return compute_Nvs2N_diff(*c)

def run_test1(config: Dict, depth_list: List[int], scaling_list: List[float],
              init_list: List[str], num_test: int = NUM_TEST, 
              num_train: int = NUM_DATA, verbose: bool = False) -> List[Dict]:
    """Perform the experiment on ODE convergence parallely

    :param config: configuration dictionary of the model
    :param depth_list: list of depth
    :param num_test: Number of test data
    :param num_train: Number of train data
    :param verbose: enable verbosity
    
    :return: list of containing the information of depth and weight difference
    """
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass
    # for depth in depth_list:
    #     for scaling in scaling_list:
    #         for init in ['iid', 'smooth']:
    results = []
    res_list = [[(copy.deepcopy(config), depth, n, scaling,
                init, np.random.randint(999999), 
                gen_data(num_train, config['dim_input']), 
                gen_data(num_test, config['dim_input']),
                verbose)
                for n in range(config['niter'])]
                for depth in depth_list
                for scaling in scaling_list
                for init in init_list]
    for item in res_list:
        res = list(map(decomp, item))
        results.extend(res)
    
                # with Pool(processes=config['n_workers']) as pool:
                #     res = pool.map(decomp, res_list)
                # res_all.extend(res)
    return results

def decomp2(c):
    return compute_Nvs0_diff(*c)

def run_test2(config: dict, depth_list: list, 
             num_test: int = NUM_TEST, num_train: int = NUM_DATA,
             verbose: bool = False) -> List[Dict]:
    res_list = [(copy.deepcopy(config), depth, n, 
                 np.random.randint(99999), 
                 gen_data(num_train, config['dim_input']), 
                 gen_data(num_test, config['dim_input']),
                 verbose)
                for n in range(config['niter']) 
                for depth in depth_list]
    pool = Pool(processes = config['n_workers'])
    res = pool.map(decomp2, res_list)
    pool.close()
    pool.join()
    #res = list(map(decomp2, res_list))
    # with Pool(processes=config['n_workers']) as pool:
    #     res = pool.map(decomp2, res_list)
    return res
    
def plot_results(results: list, scaling_order: list, init_order: List[str],
                 filepath: Optional[str] = 'figures', 
                 exp_name: Optional[str] = 'test'):
    df = pd.DataFrame(results)
    df.columns = ['depth', 'scaling', 'type', 
                  'weight_diff_max', 'weight_diff_mean']

    #df_log = df.apply(np.log)
    for norm in ['weight_diff_max', 
                    'weight_diff_mean']:
        g = sns.relplot(
            x='depth',
            y=norm,
            row = 'scaling',
            row_order = scaling_order,
            col = 'type',
            col_order = init_order,
            data=df,
            kind='line',
            facet_kws=dict(sharey=False)
        )
        for ax, col in zip(g.axes[0], init_order):
            ax.set_title(col)
        for ax in g.axes[1]:
            ax.set_title('')
            ax.set_xlabel(r'$L$')
        g.axes[0, 0].set_ylabel("scaling = 0.5")
        g.axes[1, 0].set_ylabel("scaling = 1.0")
        plt.savefig(os.path.join(filepath, f'{norm}_linear_{exp_name}.pdf'), bbox_inches='tight')
        print(f'{norm}_linear_{exp_name}.pdf saved!')
    
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    pl.seed_everything(SEED)

    model_config = config.ODE_convergence_exp_iid
    filepath_pickle = 'pickles/ODE_convergence/linear'
    pickle_name = 'diff_long.pkl'
    filepath_figures = 'figures/ODE_convergence/linear'
    exp_name = 'long'


    scaling_list = [0.5, 1.0]
    init_list = ['iid', 'smooth']
   #depth_list = np.linspace(10, 1000, num=10, dtype=int)
    depth_list = [125, 250, 500, 1000, 2000, 4000]
    #depth_list = [20, 40, 80]
    #depth_list = [20]
    resume = False
    if not resume:
        results_ODE = run_test1(
            model_config, depth_list, scaling_list, 
            init_list, verbose = False)
        
        print("Results Get!")
        
        print("start dumping the result...")
        with open(f"{filepath_pickle}/{pickle_name}", 'wb') as f:
            pickle.dump(results_ODE, f)
        print("Start plotting...")
        plot_results(results_ODE, scaling_list, init_list,
                     filepath_figures, exp_name)
        #plot_results(results_ODE, filepath)
        print("Experiment Done!")
    
    else:
        print('Start resuming previous experiment...')
        with open(f'{filepath_pickle}/{pickle_name}', 'rb') as f:
            results = pickle.load(f)
        print("Start plotting...")
        plot_results(results, scaling_list, init_list,
                     filepath_figures, exp_name)
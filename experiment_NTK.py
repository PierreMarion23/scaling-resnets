import distutils.spawn

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers
import numpy as np
import pandas as pd
import copy

from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

from torch.multiprocessing import Pool, set_start_method

from typing import Optional, Final, List, Dict
from models import SimpleResNet, FullResNet
import config
from training import multiple_fit
from utils import max_norm, mean_norm
import time
import pickle

NUM_DATA: Final[int] = 60

class MyCallback(Callback):
    def __init__(self, train_data, width, logger):
        super().__init__()
        self.epoch_count_ = 0
        self.train_data = train_data
        self.width = width
        self.logger = logger

    def on_train_epoch_end(self, trainer, model):
        self.epoch_count_+=1
        self.logger.log_metrics(
            {'epochs': self.epoch_count_, 
             'kernel': compute_kernel(model, self.train_data, self.width)} 
        )

def gen_data(N: int = NUM_DATA):
    return torch.utils.data.TensorDataset(torch.randn(N, 1), torch.randn(N, 1))

def compute_kernel(model: pl.LightningModule, 
                   data: torch.utils.data.Dataset, width: int,
                   N: int = NUM_DATA, verbose: bool = True) -> torch.Tensor:
    vec: List[torch.Tensor] = []
    res = torch.zeros((N, N))
    for x, y in data:
        output = model.forward(x)
        output.backward(retain_graph = True)
        layer_grads = []
        for param in list(model.parameters()):
            print(param.grad)
            layer_grads.append(param.grad.reshape((-1, width)).detach().clone())
            param.grad.zero_()
        vec.append(torch.vstack(layer_grads).reshape((-1, )))
    for i in range(len(vec)):
        for j in range(len(vec)):
            res[i, j] = torch.dot(vec[i], vec[j])
    return res

def fit(config: Dict, depth: int, 
             data, verbose: bool=False) -> Dict:
    model_config = config['model-config']
    model_config['depth'] = depth
    model = FullResNet(config['dim_input'],
                       config['nb_classes'], **model_config)
    gpu = 1 if torch.cuda.is_available() else 0
    logger = loggers.CSVLogger(
                save_dir='lightning_logs',
                name='NTK_exp',
                )
    trainer = pl.Trainer(
            gpus=gpu,
            max_epochs=config['epochs'],
            enable_checkpointing=False,
            logger=logger,
            callbacks=[MyCallback(data, model_config['width'], logger)],
            enable_progress_bar=verbose,
            enable_model_summary=verbose
        )
    train_dl = DataLoader(data)
    trainer.fit(model, train_dl)

def run_test(config: Dict, depth_list: List[int], 
             N: int = 60, verbose: bool = False):
    res_list = []
    for depth in depth_list:
        print(f"depth: {depth}")
        for n in range(config['niter']):
            print(f"round {n} starts")
            data = gen_data(N)
            res = fit(config, depth, data, verbose)
            res_list.append(res)
    print("Test finished")
    return res_list

def plot_results(results: List[Dict], filepath: Optional[str] = 'figures',
                 exp_name: Optional[str] = None):
    print("Start plotting...")
    df = pd.DataFrame(results)
    df.columns = ['depth', 'DeltaK']
    g = sns.relplot(
        x = 'depth',
        y = 'DeltaK',
        data = df,
        kind = 'line',

    )
    g.set_xlabel(r"$L$")
    plt.savefig(f"{filepath}/NTK_DeltaK_{exp_name}.pdf", bbox_inches='tight')
    #plt.show()


if __name__ == "__main__":
    
    model_config = config.NTK_exp
    fit(model_config, 20, gen_data(), True)
    
    # depth_list = [20, 40, 60]
    # pickle_filepath = 'pickles/NTK'
    # pickle_name = "kernel_diff.pkl"
    # figures_filepath = 'figures/NTK'
    # exp_name = "test"
    # resume = False
    # if not resume:
    #     results = run_test(model_config, depth_list)
    #     os.makedirs(pickle_filepath, exist_ok=True)
    #     with open(f'{pickle_filepath}/{pickle_name}', 'wb') as f:
    #         pickle.dump(results, f)
    #     os.makedirs(figures_filepath, exist_ok=True)
        
    #     plot_results(results, figures_filepath, exp_name)
    
    # else:
    #     print("Resuming the previous experiment")
    #     with open(f"{pickle_filepath}/{pickle_name}", 'rb') as f:
    #         results = pickle.load(f)
    #     print("Start plotting...")
    #     plot_results(results, figures_filepath, exp_name)



    

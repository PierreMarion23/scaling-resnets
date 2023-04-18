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
from models import FullResNet
import config
from training import multiple_fit
from utils import max_norm, mean_norm
import time
import pickle

NUM_DATA: Final[int] = 60

class MyCallback(Callback):
    def __init__(self, train_dl, width):
        self.epoch_count_ = 0
        self.train_dl = train_dl
        self.width = width

    def on_train_epoch_end(self, trainer, model):
        self.epoch_count_+=1
        self.logger.log_metrics(
            {'epochs': self.epoch_count_, 
             'kernel': compute_kernel(model, self.train_dl, self.width)} 
        )

def gen_data(N: int = NUM_DATA, shift: float = 10.):
    n = N //3
    x0 = torch.utils.data.TensorDataset(torch.rand(n, 1), 
                                        F.one_hot(torch.full((n, ), 0),
                                                  num_classes = 3).float())
    x1 = torch.utils.data.TensorDataset(shift+torch.rand(n, 1), 
                                        F.one_hot(torch.full((n, ), 1),
                                                  num_classes = 3).float())
    x2 = torch.utils.data.TensorDataset(-shift+torch.rand(n, 1), 
                                        F.one_hot(torch.full((n, ), 2),
                                                  num_classes = 3).float())
    return torch.utils.data.ConcatDataset([x0, x1, x2])

def compute_kernel(model: pl.LightningModule, 
                   data: torch.utils.data.Dataset, width: int,
                   N: int = NUM_DATA, verbose: bool = True) -> torch.Tensor:
    vec: List[torch.Tensor] = []
    res = torch.zeros((N, N))
    model.train()
    for x, y in data:
        output = model.forward(x)
        output.backward(retain_graph = True)
        layer_grads = []
        for name, param in model.named_parameters():
            print(name, param)
            layer_grads.append(param.grad.reshape((-1, width)).detach().clone())
            param.grad.zero_()
        # for o, i in zip(model.outer_weights, model.inner_weights):
        #     layer_grads.append(o.weight.grad.detach().clone())
        #     layer_grads.append(i.weight.grad.detach().clone())
        #     o.weight.grad.zero_()
        #     i.weight.grad.zero_()
        # layer_grads.append(model.init.weight.grad.detach().clone())
        # layer_grads.append(model.final.weight.grad.detach().clone())
        # model.init.weight.grad.zero_()
        # model.final.weight.grad.zero_()
        vec.append(torch.vstack(layer_grads).reshape((-1, )))
    for i in range(len(vec)):
        for j in range(len(vec)):
            res[i, j] = torch.dot(vec[i], vec[j])
    return res

def compute_diff(config: Dict, depth: int, 
             data, verbose: bool=False) -> Dict:
    model_config = config['model-config']
    model_config['depth'] = depth
    model = FullResNet(config['dim_input'],
                       config['nb_classes'], **model_config)
    kernel_init = compute_kernel(model, data, model_config['width'])
    gpu = 1 if torch.cuda.is_available() else 0
    # logger = loggers.CSVLogger(
    #             save_dir='lightning_logs',
    #             name='NTK_exp',
    #             )
    trainer = pl.Trainer(
            gpus=gpu,
            max_epochs=config['epochs'],
            enable_checkpointing=False,
            enable_progress_bar=verbose,
            enable_model_summary=verbose
        )
    train_dl = DataLoader(data, num_workers = 32)
    trainer.fit(model, train_dl)
    kernel_final = compute_kernel(model, data, model_config['width'])
    deltaK = torch.norm(kernel_final-kernel_init).item() / \
                torch.norm(kernel_init).item()
    return {'depth': depth, 'DeltaK': deltaK}

def run_test(config: Dict, depth_list: List[int], 
             N: int = 60, verbose: bool = False):
    res_list = []
    for depth in depth_list:
        print(f"depth: {depth}")
        for n in range(config['niter']):
            print(f"round {n} starts")
            data = gen_data(N)
            res = compute_diff(config, depth, data, verbose)
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
    
    depth_list = [20, 40, 60]
    pickle_filepath = 'pickles/NTK'
    pickle_name = "kernel_diff.pkl"
    figures_filepath = 'figures/NTK'
    exp_name = "test"
    resume = False
    if not resume:
        results = run_test(model_config, depth_list)
        os.makedirs(pickle_filepath, exist_ok=True)
        with open(f'{pickle_filepath}/{pickle_name}', 'wb') as f:
            pickle.dump(results, f)
        os.makedirs(figures_filepath, exist_ok=True)
        
        plot_results(results, figures_filepath, exp_name)
    
    else:
        print("Resuming the previous experiment")
        with open(f"{pickle_filepath}/{pickle_name}", 'rb') as f:
            results = pickle.load(f)
        print("Start plotting...")
        plot_results(results, figures_filepath, exp_name)




    

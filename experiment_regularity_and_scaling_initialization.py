import distutils.spawn
import os
from typing import Optional

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import config
from models import FullResNet
import utils


sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def run_experiment(config: dict, grid_beta: list, grid_reg: list) -> list:
    """ Loop over a grid of initializations and scaling values, compute ratios
    between norms of the output and the input, as well as their respective
    gradients.

    :param config: configuration of the experiment
    :param grid_beta: all values of scaling to loop over
    :param grid_reg: regularity of the process used to initialize the weights
    of the network
    :return:
    """

    results = []
    for reg in grid_reg:
        print(reg)
        for k in range(config['niter_reg']):
            model_config = config['model-config']
            model_config['regularity']['type'] = 'fbm'
            model_config['regularity']['value'] = reg
            model = FullResNet(
                config['dim_input'], config['nb_classes'], **model_config)

            for beta in grid_beta:
                model.reset_scaling(beta)
                for j in range(config['niter_beta']):

                    x0 = torch.rand((1, config['dim_input']))
                    target = torch.rand((1,))

                    h_0 = model.init(x0)
                    h_L = model.forward_hidden_state(h_0)
                    output = model.final(h_L)

                    h_0.retain_grad()
                    h_L.retain_grad()

                    loss = torch.norm(output - target) ** 2
                    loss.backward()

                    h_0_grad = h_0.grad
                    h_L_grad = h_L.grad

                    results.append({
                        'beta': beta,
                        'regularity': reg,
                        'hidden_state_difference': float(
                            torch.norm(h_L - h_0) / torch.norm(h_0)),
                        'gradient_difference': float(
                            torch.norm(h_L_grad - h_0_grad) / torch.norm(
                                h_L_grad)),
                        })
    return results


def plot_results(results: list, filepath: Optional[str] = 'figures'):
    """Plot heatmaps which describes the hidden state and gradient norms,
    as a function of scaling and initialization regularity.
    See Figure 7 of the paper.

    :param results: list of results
    :return:
    """
    df = pd.DataFrame(results)
    df.columns = ['scaling', 'regularity', 'hidden_state_difference',
                  'gradient_difference']
    df['log_hidden_state_difference'] = np.log10(df['hidden_state_difference'])
    df['log_gradients_difference'] = np.log10(df['gradient_difference'])

    df1 = df.pivot_table(index='scaling', columns='regularity',
                         values='log_hidden_state_difference', dropna=False)
    df1.index = np.round(df1.index.astype(float), 2)
    df1.columns = np.round(df1.columns.astype(float), 2)
    df1.replace(np.inf, np.nan, inplace=True)
    df1.fillna(16, inplace=True)
    sns.heatmap(df1[::-1], vmin=-1, vmax=1, center=0, xticklabels=5,
                yticklabels=6, square=True)
    plt.savefig(
        os.path.join(filepath, 'heatmap-scaling-regularity-output.pdf'),
        bbox_inches='tight')
    plt.show()

    df2 = df.pivot_table(index='scaling', columns='regularity',
                         values='log_gradients_difference', dropna=False)
    df2.index = np.round(df2.index.astype(float), 2)
    df2.columns = np.round(df2.columns.astype(float), 2)
    df2.replace(np.inf, np.nan, inplace=True)
    df2.fillna(16, inplace=True)
    sns.heatmap(df2[::-1], vmin=-1, vmax=1, center=0, xticklabels=5,
                yticklabels=6, square=True)
    plt.savefig(
        os.path.join(filepath, 'heatmap-scaling-regularity-gradient.pdf'),
        bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # Plot examples of fractional Brownian Motion - Figure 6 of the paper
    grid_H = [0.2, 0.5, 0.8]
    filepath = 'figures/regularity_and_scaling_initialization'
    os.makedirs(filepath, exist_ok=True)
    for hurst in grid_H:
        path = utils.generate_fbm(1000, hurst)[0]
        plt.plot(path)
        plt.savefig(
            os.path.join(filepath, 'fbm_examples-{:.1f}.pdf'.format(hurst)),
            bbox_inches='tight')
        plt.show()

    # Run the experiments to loop over various initializations and scalings
    # Figure 7 of the paper
    config_heatmap = config.scaling_regularity_initialization_exp

    grid_beta = list(np.linspace(0, 1.3, 70))
    grid_reg = list(np.linspace(0.05, 0.97, 51))

    results = run_experiment(config_heatmap, grid_beta, grid_reg)
    plot_results(results, filepath)

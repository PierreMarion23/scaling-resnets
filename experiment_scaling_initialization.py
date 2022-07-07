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

sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def run_experiment(
        config: dict, grid_beta: list, grid_depth: list) -> list:
    """ Loop over a grid of depths and scaling values, compare the norm of the
    last layer values to the initial one, as well as their respective
    gradients.

    :param config: configuration of the experiment
    :param grid_beta: all values of scaling to loop over
    :param grid_depth: all depths of the ResNet to loop over
    :return:
    """
    results = []
    for beta in grid_beta:
        for depth in grid_depth:
            print(depth)
            for k in range(config['niter']):
                if config['niter'] > 10**3 and k % 10 == 0:
                    print(k)
                model_config = config['model-config']
                model_config['scaling_beta'] = beta
                model_config['depth'] = depth

                x0 = torch.rand((1, config['dim_input']))
                target = torch.rand((1,))
                model = FullResNet(
                    config['dim_input'], config['nb_classes'],
                    **model_config)

                h_0 = model.init(x0)
                h_L = model.forward_hidden_state(h_0)
                output = model.final(h_L)

                h_0.retain_grad()
                h_L.retain_grad()

                # model.train()
                loss = torch.norm(output - target)**2
                loss.backward()

                h_0_grad = h_0.grad
                h_L_grad = h_L.grad

                results.append({
                    'depth': depth,
                    'beta': beta,
                    'hidden_state_ratio': float(
                        torch.norm(h_L) / torch.norm(h_0)),
                    'hidden_state_difference': float(
                        torch.norm(h_L - h_0) / torch.norm(h_0)),
                    'gradient_ratio': float(
                        torch.norm(h_0_grad) / torch.norm(h_L_grad)),
                    'gradient_difference': float(
                        torch.norm(h_L_grad - h_0_grad) / torch.norm(h_L_grad))
                })
    return results


def plot_histogram(results: list, filepath: Optional[str] = 'figures'):
    """Plot an histogram of ratios between initial and final norms, 
    and initial and final gradients. The depth L is fixed.
    See Figure 2 of the paper.

    :param results: list of results
    :param filepath: path to the folder where the figures should be saved
    :return:
    """
    df = pd.DataFrame(results)
    df.columns = ['depth', r'$\beta$', 'hidden_state_ratio',
                  'hidden_state_difference', 'gradient_ratio',
                  'gradient_difference']

    g = sns.histplot(x='hidden_state_ratio', data=df)
    g.set_xlabel(r'$\|h_L\| / \|h_0\|$')
    plt.savefig(os.path.join(filepath, 'hist_norm_initialization.pdf'), 
                bbox_inches='tight')
    plt.show()

    g = sns.histplot(x='gradient_ratio', data=df)
    g.set_xlabel(
        r'$ \|\frac{\partial \mathcal{L}}{\partial h_0}\| / '
        r'\|\frac{\partial \mathcal{L}}{\partial h_L}\|$')
    plt.savefig(os.path.join(filepath, 'hist_gradient_initialization.pdf'),
                bbox_inches='tight')
    plt.show()


def plot_results(results: list, col_order: list, filepath: Optional[str] = 'figures'):
    """ Plot relative ratios between last and first hidden state norms and gradient norms,
    as a function of the depth L.
    See Figures 1, 3, 4, 5 of the paper.

    :param results: list of results
    :param col_order: grid of beta values for plotting
    :param filepath: path to the folder where the figures should be saved
    :return:
    """
    df = pd.DataFrame(results)
    df.columns = ['depth', r'$\beta$', 'hidden_state_ratio',
                  'hidden_state_difference', 'gradient_ratio',
                  'gradient_difference']

    g = sns.relplot(
        x='depth',
        y='hidden_state_difference',
        col=r'$\beta$',
        col_wrap=3,
        col_order=col_order,
        data=df,
        kind='line',
        facet_kws=dict(sharey=False)
    )

    g.axes[0].set_ylabel('')
    g.axes[0].set_xlabel(r'$L$')
    g.axes[1].set_xlabel(r'$L$')
    g.axes[2].set_xlabel(r'$L$')
    plt.savefig(os.path.join(filepath, 'norm_output_initialization.pdf'), bbox_inches='tight')
    plt.show()

    g = sns.relplot(
        x='depth',
        y='gradient_difference',
        col=r'$\beta$',
        col_wrap=3,
        col_order=col_order,
        data=df,
        kind='line',
        facet_kws=dict(sharey=False)
    )

    g.axes[0].set_ylabel('')
    g.axes[0].set_xlabel(r'$L$')
    g.axes[1].set_xlabel(r'$L$')
    g.axes[2].set_xlabel(r'$L$')
    plt.savefig(
        os.path.join(filepath, 'norm_gradient_initialization.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # Experiment with i.i.d. initialization - Figures 1 and 3 of the paper
    config_iid = config.scaling_initialization_exp_iid
    filepath = 'figures/scaling_initialization/iid'
    os.makedirs(filepath, exist_ok=True)

    grid_depth = np.linspace(10, 1e3, num=10, dtype=int)
    grid_beta = [1.0, 0.25, 0.5]

    results_iid = run_experiment(
        config_iid, grid_beta, grid_depth)

    plot_results(results_iid, grid_beta, filepath)

    # Distribution of norms for i.i.d. initialization - Figure 2 of the paper
    config_hist = config.histogram_initialization_exp
    results_hist = run_experiment(
        config_hist, [config_hist['model-config']['scaling_beta']],
        [config_hist['model-config']['depth']])
    plot_histogram(results_hist, filepath)

    # Experiment with smooth initialization - Figures 4 and 5 of the paper
    config_smooth = config.scaling_initialization_exp_smooth
    filepath = 'figures/scaling_initialization/smooth'
    os.makedirs(filepath, exist_ok=True)

    grid_beta = [2., 0.5, 1.]
    results_smooth = run_experiment(
        config_smooth, grid_beta, grid_depth)
    plot_results(results_smooth, grid_beta, filepath)

import distutils.spawn
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

    :param config:
    :param grid_beta:
    :param grid_depth:
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


def plot_histogram(results: list):
    """ Reproduces Figure 2 of the paper

    :param results: list of results
    :return:
    """
    df = pd.DataFrame(results)
    df.columns = ['depth', r'$\beta$', 'hidden_state_ratio',
                  'hidden_state_difference', 'gradient_ratio',
                  'gradient_difference']

    g = sns.histplot(x='hidden_state_ratio', data=df)
    g.set_xlabel(r'$\|h_L\| / \|h_0\|$')
    plt.savefig('figures/hist_norm_initialization.pdf', bbox_inches='tight')
    plt.show()

    g = sns.histplot(x='gradient_ratio', data=df)
    g.set_xlabel(
        r'$ \|\frac{\partial \mathcal{L}}{\partial h_0}\| / '
        r'\|\frac{\partial \mathcal{L}}{\partial h_L}\|$')
    plt.savefig('figures/hist_gradient_initialization.pdf',
                bbox_inches='tight')
    plt.show()


def plot_results(results: list, col_order: list):
    """ Reproduce Figures 1 and 3 of the paper

    :param results: list of results
    :param col_order: grid of beta values for plotting
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
    plt.savefig('figures/norm_output_initialization.pdf', bbox_inches='tight')
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
        'figures/norm_gradient_initialization.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # Experiment with i.i.d. initialization
    config_iid = config.scaling_initialization_exp
    config_iid['model-config']['regularity'] = {'type': 'iid'}

    grid_depth = np.linspace(10, 1e3, num=10, dtype=int)
    grid_beta = [1.0, 0.25, 0.5]

    results_iid = run_experiment(
        config_iid, grid_beta, grid_depth)

    plot_results(results_iid, grid_beta)

    # Distribution of norms when beta=1/2
    config_hist = config.histogram_initialization_exp
    results_hist = run_experiment(
        config_hist, [config_hist['model-config']['scaling_beta']],
        [config_hist['model-config']['depth']])
    plot_histogram(results_hist)

    # Experiment with smooth initialization
    config_smooth = config.scaling_initialization_exp
    config_smooth['model-config']['regularity'] = {
        'type': 'rbf', 'value': 0.01}

    grid_beta = [2., 0.5, 1.]
    results_smooth = run_experiment(
        config_smooth, grid_beta, grid_depth)
    plot_results(results_smooth, grid_beta)



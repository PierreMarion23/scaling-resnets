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


def plot_results(results: list, col_order: list):
    df = pd.DataFrame(results)
    df.columns = ['depth', r'$\beta$', 'hidden_state_ratio',
                  'hidden_state_difference', 'gradient_ratio',
                  'gradient_difference']

    print('Evolution of the norm of the output as a function of L for '
          'different values of beta')
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

    print('Evolution of the norm of the gradients as a function of L for '
          'different values of beta')
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
    config_iid = config.scaling_initialization_exp
    config_iid['model-config']['regularity'] = {'type': 'iid'}

    grid_depth = np.linspace(10, 1e3, num=10, dtype=int)
    grid_beta = [1.0, 0.25, 0.5]

    # Experiment with i.i.d. initialization
    results_iid = run_experiment(
        config_iid, grid_beta, grid_depth)

    plot_results(results_iid, grid_beta)

    # Experiment with smooth initialization
    config_smooth = config.scaling_initialization_exp
    config_smooth['model-config']['regularity'] = {'type': 'rbf', 'value': 0.01}

    grid_beta = [2., 0.5, 1.]
    results_smooth = run_experiment(
        config_smooth, grid_beta, grid_depth)
    plot_results(results_smooth, grid_beta)



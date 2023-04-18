import distutils.spawn
import os
from typing import Optional

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import utils
import config
import pickle 
from models import FullResNet
from typing import Final

SEED: Final[int] = 42
sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def run_experiment(
        config: dict, grid_scaling: list, grid_depth: list, cov) -> list:
    """ Loop over a grid of depths and scaling values, compare the norm of the
    last layer values to the initial one, as well as their respective
    gradients.

    :param config: configuration of the experiment
    :param grid_scaling: all values of scaling to loop over
    :param grid_depth: all depths of the ResNet to loop over
    :return:
    """
    results = []
    for scaling in grid_scaling:
        print(f"beta: {scaling}")
        for depth in grid_depth:
            print(depth)
            for k in range(config['niter']):
                if config['niter'] > 10**3 and k % 10 == 0:
                    print(k)
                model_config = config['model-config']
                model_config['scaling'] = scaling
                model_config['depth'] = depth
                model_config['cov'] = cov

                x0 = torch.rand((1, config['dim_input']))
                target = torch.rand((1,))
                model = FullResNet(
                    config['dim_input'], config['nb_classes'],
                    **model_config)

                print(f"round {k}")
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
                    'scaling': scaling,
                    'hidden_state_ratio': float(
                        torch.norm(h_L) / torch.norm(h_0)),
                    'hidden_state_difference': float(
                        torch.norm(h_L - h_0) / torch.norm(h_0)),
                    'gradient_ratio': float(
                        torch.norm(h_0_grad) / torch.norm(h_L_grad)),
                    'gradient_difference': float(
                        torch.norm(h_L_grad - h_0_grad) / torch.norm(h_L_grad))
                })
                print(f"round {k} finished")
    return results


def plot_results(results: list, col_order: list, filepath: Optional[str] = 'figures'):
    """ Plot relative ratios between last and first hidden state norms and gradient norms,
    as a function of the depth L.
    See Figures 1, 3, 4, 5 of the paper.

    :param results: list of results
    :param col_order: grid of scaling values for plotting
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
    # Experiment with smooth initialization - Figures 4 and 5 of the paper
    config_smooth = config.scaling_initialization_exp_smooth_with_corr
    filepath = 'figures/scaling_initialization/smooth'
    os.makedirs(filepath, exist_ok=True)
    cov = 1/config_smooth['model-config']['width'] * \
        utils.create_cov_matrix(config_smooth['model-config']['width'], seed=SEED)
    pickle_filepath = 'pickles/scaling_initialization_cov_matrices'
    os.makedirs(pickle_filepath, exist_ok=True)
    with open("pickles/scaling_initialization_cov_matrices/rbf_cov.pkl", "wb") as f:
        pickle.dump(cov, f)
    print(cov)
    grid_scaling = [2., 0.5, 1.]
    grid_depth = np.linspace(10, 1e3, num=10, dtype=int)
    results_smooth = run_experiment(
        config_smooth, grid_scaling, grid_depth, cov = cov)
    plot_results(results_smooth, grid_scaling, filepath)

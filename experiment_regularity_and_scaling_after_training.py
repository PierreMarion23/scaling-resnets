import distutils.spawn

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
import training

sns.set(font_scale=1.5)
if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def plot_heatmap(results: list):
    """Plot heatmaps which describes the performance,
    as a function of scaling and initialization regularity.
    See Figure 9 of the paper.

    :param results: list of results
    :return:
    """
    df = pd.DataFrame.from_dict(results)

    df2 = pd.pivot_table(df, index='scaling', columns='regularity',
                         values='accuracy', aggfunc=np.max)
    df2.index = np.round(df2.index, 2)
    df2.columns = np.round(df2.columns, 2)

    sns.heatmap(df2[::-1], vmin=0, vmax=1, center=0.5, xticklabels=1,
                yticklabels=1, square=True)
    plt.savefig('figures/perf-weights-regularity.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    grid_lr = [0.0001, 0.001, 0.01, 0.1, 1.]
    grid_regularity = np.linspace(0.1, 0.9, 10)
    grid_beta = np.linspace(0.1, 0.9, 10)
    exp_config = config.perf_weights_regularity

    for dataset in ['MNIST', 'CIFAR10']:
        exp_config['dataset'] = dataset
        training.fit_parallel(exp_config, grid_lr, grid_regularity, grid_beta)
        exp_name = exp_config['name'].replace('dataset', dataset)
        results = training.get_results(exp_name)
        plot_heatmap(results)

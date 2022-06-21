import distutils.spawn
import glob
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns


sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def get_results(exp_name: str) -> list:
    """Read the results of test accuracy saved after execution of the main
    file.

    :param exp_name: name of the configuration
    :return: list of results
    """
    results = {'accuracy': [], 'regularity': [], 'lr': [], 'scaling': []}
    for directory in glob.glob(os.path.join('results', exp_name, '*')):
        with open(os.path.join(directory, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
            results['regularity'].append(
                config['model-config']['regularity']['value'])
            results['lr'].append(config['model-config']['lr'])
            results['scaling'].append(config['model-config']['scaling_beta'])
        with open(os.path.join(directory, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)
            results['accuracy'].append(metrics['test_accuracy'])

    return results


def plot_heatmap(results: list):
    """Reproduces Figure 9 of the paper.

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
    dataset = 'MNIST'
    exp_name = f'perf-weights-regularity-{dataset}'
    results = get_results(exp_name)
    plot_heatmap(results)


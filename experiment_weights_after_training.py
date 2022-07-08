import distutils.spawn
import os
from typing import Optional

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
import training

sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def run_experiment(exp_config: dict, filepath: Optional[str] = 'figures'):
    """Train a model and plot a random weight as a function
    of the layer index.

    :param exp_config: configuration of the experiment
    :param filepath: path to the folder where the figures should be saved
    :return:
    """
    model = training.fit(exp_config, verbose=True)
    random_index = [np.random.randint(exp_config['model-config']['width']),
                    np.random.randint(exp_config['model-config']['width'])]

    weight_example = np.array([
        model.outer_weights[k].weight[random_index[0],
                                      random_index[1]].detach().numpy()
        for k in range(exp_config['model-config']['depth'])])

    plt.plot(weight_example)
    plt.savefig(
        os.path.join(filepath, "example_weight_%.1f_%s.pdf" % 
        (exp_config['model-config']['scaling'], exp_config['model-config']['regularity']['type'])), 
        bbox_inches='tight'
    )
    plt.show()


if __name__ == '__main__':
    exp_config = config.perf_weights_regularity
    grid_scaling_reg = [{'scaling': 1, 'regularity': {'type': 'rbf', 'value': 0.01}},
                   {'scaling': 1, 'regularity': {'type': 'iid'}},
                   {'scaling': 0.5, 'regularity': {'type': 'iid'}}
                   ]
    filepath = 'figures/weights_after_training'
    os.makedirs(filepath, exist_ok=True)
    for scaling_reg in grid_scaling_reg:
        exp_config['model-config'] = scaling_reg
        run_experiment(exp_config, filepath)

import distutils.spawn
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
import main

sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def run_experiment(config):
    model = main.fit(config, verbose=True)
    random_index = [np.random.randint(config['model-config']['width']),
                    np.random.randint(config['model-config']['width'])]

    weight_example = np.array([
        model.outer_weights[k].weight[random_index[0],
                                      random_index[1]].detach().numpy()
        for k in range(config['model-config']['depth'])])

    plt.plot(weight_example)
    plt.savefig(f"figures/example_weight_beta.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    config_exp = config.perf_weights_regularity
    grid_beta_reg = [ #{'beta': 1, 'regularity': {'type': 'rbf', 'value': 0.01}},
                   {'beta': 1, 'regularity': {'type': 'iid'}},
                   #{'beta': 0.5, 'regularity': {'type': 'iid'}}
                   ]
    for beta_reg in grid_beta_reg:
        config_exp['model-config']['scaling_beta'] = beta_reg['beta']
        config_exp['model-config']['regularity'] = beta_reg['regularity']
        run_experiment(config_exp)





"""Combining the result obtained from the experiment of ODE_convergence linear 
and non linear by plotting them in a same graph.
"""
import distutils.spawn

import os
import numpy as np
import pandas as pd

from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, List
import pickle


sns.set(font_scale=1.5)

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def plotting(dfm: pd.DataFrame, init_order: List[str],
             activ_order: List[str], scaling_order: List[float],
             log: bool=False):
    
    sns.set_palette("bright", 4)
    values = 'values_log' if log else 'values'
    xlim = (10, 1000) if log else (0, 1000)
    g = sns.relplot(x='depth',
                    y=values,
                    style='vars',
                    style_order = ['mean', 'max'],
                    hue='activ',
                    hue_order = activ_order,
                    row='scaling',
                    row_order=scaling_order,
                    col='type',
                    col_order=init_order,
                    data=dfm,
                    kind='line',
                    facet_kws=dict(sharey=False))
        # new_labels = ['label 1', 'label 2',]
        # for t, l in zip(leg.texts, new_labels):
        #     t.set_text(l)
    if log: plt.xscale("log")
    for ax, col in zip(g.axes[0], init_order):
        ax.set_title(col)
        ax.set_xlabel('')
    for ax in g.axes[1]:
        ax.set_title('')
        ax.set_xlabel(r'$L$')
    g.axes[0, 0].set_ylabel(r'$\alpha$ = 0.5')
    g.axes[1, 0].set_ylabel(r'$\alpha$ = 1.0')
    g._legend.texts[0].set_text('Activation')
    g._legend.texts[5].set_text('Norm')

if __name__=="__main__":
    filepath = "figures/ODE_convergence/combined"
    os.makedirs(filepath, exist_ok=True)
    ex = ""
    exp_name = "combined"
    with open('pickles/ODE_convergence/linear/diff_mod3.pkl', 'rb') as f:
        results_linear = pickle.load(f)

    with open('pickles/ODE_convergence/non_linear/diff_10-100030iter.pkl', 'rb') as f:
        results_non = pickle.load(f)

    df_non = pd.DataFrame(results_non)
    df_non.columns = ['depth', 'activ', 'scaling',
                  'type', 'weight_diff_max', 'weight_diff_mean',
                  'init_diff_max', 'init_diff_mean', 'max_weight2N']
    df_non_sel = df_non[['depth', 'activ', 'scaling', 'type', 
                        'weight_diff_max',
                        'weight_diff_mean']]
    df_lin = pd.DataFrame(results_linear)
    df_lin.columns = ['depth', 'scaling', 'type', 
                        'weight_diff_max', 'weight_diff_mean']
    df = pd.concat([df_non_sel,
                    df_lin.assign(activ='linear')])
    dfm = df.melt(['depth', 'type', 'scaling', 'activ'],
                  var_name='vars', value_name='values')
    dfm['vars'] = dfm['vars'].replace(['weight_diff_max', 'weight_diff_mean'], 
                                      ['max', 'mean'])
    dfm['type'] = dfm['type'].replace(['smooth'], 'rbf')
    dfm['depth_log'] = np.log(dfm['depth'])
    dfm['values_log'] = np.log(dfm['values'])

    scaling_order = [0.5, 1.0]
    activ_order = ['ReLU', 'Sigmoid', 'Tanh']
    init_order = ['iid', 'rbf']
    activ_order = ['linear', 'Sigmoid', 'ReLU', 'Tanh']
    plotting(dfm, init_order, activ_order, scaling_order)
    plt.savefig(
        os.path.join(filepath, f'diff_{exp_name}.pdf'), bbox_inches='tight')
    print(f'diff_{exp_name}.pdf saved!')
    
    plotting(dfm, init_order, activ_order, scaling_order, log = True)
    plt.savefig(
        os.path.join(filepath, f'diff_log_{exp_name}.pdf'), bbox_inches='tight')
    print(f'diff_log_{exp_name}.pdf saved!')
        
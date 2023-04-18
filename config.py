
scaling_initialization_exp_iid = {
    'name': 'scaling_initialization_exp',
    'model': 'FullResnet',
    'model-config': {
        'width': 40,
        'activation': 'ReLU',
        'regularity': {'type': 'iid'}
    },
    'niter': 50,
    'dim_input': 64,
    'nb_classes': 1,
}

scaling_initialization_exp_iid_with_cov ={
    'name': 'scaling_initialization_with_cov',
    'model': 'FullResnet',
    'model-config':{
        'width': 40,
        'activation': 'ReLU',
        'regularity': {'type': 'iid_with_cov',
                       'with_cov': True}
    },
    'niter': 30,
    'dim_input': 64,
    'nb_classes': 1
}

scaling_initialization_exp_iid_leakyReLU = {
    'name': 'scaling_initialization_exp_leakyReLU',
    'model': 'FullResnet',
    'model-config': {
        'width': 40,
        'activation': 'LeakyReLU',
        'negative_slope': 0.8,
        'regularity': {'type': 'iid'},
    },
    'niter': 50,
    'dim_input': 64,
    'nb_classes': 1
}

scaling_initialization_exp_iid_factor_in_distro = {
    'name': 'scaling_initialization_factor_in_distro',
    'model': 'FullResnet',
    'model-config': {
        'width': 40,
        'activation': 'ReLU',
        'regularity': {'type': 'iid'}
    },
    'in_distro': True,
    'niter': 50,
    'dim_input': 64,
    'nb_classes': 1
}

scaling_initialization_exp_smooth = {
    'name': 'scaling_initialization_exp',
    'model': 'FullResnet',
    'model-config': {
        'width': 40,
        'activation': 'ReLU',
        'regularity': {
            'type': 'rbf',
            'value': 0.1
        }
    },
    'niter': 50,
    'dim_input': 64,
    'nb_classes': 1,
}

scaling_initialization_exp_smooth_with_corr = {
    'name': 'scaling_initialization_exp_with_corr',
    'model': 'FullResnet',
    'model-config':{
        'width': 40,
        'activation': 'ReLU',
        'regularity': {
            'type': 'rbf_with_corr',
            'value': 0.1
        }
    },
    'niter': 50,
    'dim_input': 64,
    'nb_classes': 1,
}

histogram_initialization_exp = {
    'name': 'histogram_initialization_exp',
    'model': 'FullResnet',
    'model-config': {
        'width': 100,
        'depth': 10 ** 3,
        'scaling': 0.5,
        'activation': 'ReLU',
        'regularity': {'type': 'iid'}
    },
    'niter': 10 ** 4,
    'dim_input': 64,
    'nb_classes': 1,
}

scaling_regularity_initialization_exp = {
    'name': 'scaling_regularity_initialization_exp',
    'model': 'FullResnet',
    'model-config': {
        'width': 40,
        'depth': 1000,
        'scaling': 0.5,
        'activation': 'ReLU',
        'regularity': {}
    },
    'niter_reg': 5,
    'niter_scaling': 10,
    'dim_input': 64,
    'nb_classes': 1,
}

scaling_regularity_initilialization_exp_volterra = {
    'name': 'scaling_regularity_initialization_exp_volterra',
    'model': 'FullResnet',
    'model-config': {
        'width': 40,
        'depth': 1000,
        'scaling': 0.5,
        'activation': 'ReLU',
        'regularity': {}
    },
    'niter_reg': 5,
    'niter_scaling': 10,
    'dim_input': 64,
    'nb_classes': 1
}

ODE_convergence_exp_iid = {
    'name': 'ode_convergence_exp',
    'model': 'LinearResNet',
    'model-config': {
        'width': 40,
        'lr': 0.01,
        'scaling': 1,
        'activation': 'ReLU',
        'step_lr': 1000,
        'loss': 'MSELoss',
        'regularity': {'type': 'iid',
                       'value': 0.1}
    },
    'niter': 20,
    'dim_input': 64,
    'epochs': 5,
    'n_workers':5,
    'nb_classes': 1
}


ODE_convergence_exp_smooth = {
    'name': 'ode_convergence_exp',
    'model': 'LinearResNet',
    'model-config': {
        'width': 40,
        'lr': 0.01,
        'scaling': 1,
        'activation': 'ReLU',
        'step_lr': 1000,
        'loss': 'MSELoss',
        'regularity': {'type': 'smooth',
                       'value': 0.1}
    },
    'niter': 30,
    'dim_input': 64,
    'epochs': 5,
    'n_workers':5,
    'nb_classes': 1
}

ODE_convergence_exp_non_linear = {
    'name': 'ode_convergence_exp',
    'model': 'FullResNet',
    'model-config': {
        'width': 40,
        'lr': 0.01,
        'scaling': 1,
        'activation': 'ReLU',
        'step_lr': 1000,
        'loss': 'MSELoss',
        'regularity': {'type': "rbf",
                       'value': 0.1}
    },
    'niter': 30,
    'dim_input': 64,
    'epochs': 5,
    'n_workers':5,
    'nb_classes': 1
}

NTK_exp = {
    'name': 'ntk_exp',
    'model': 'FullResNet',
    'model-config':{
        'width': 20,
        'lr': 0.01,
        'scaling': 0.5,
        'activation': 'ReLU',
        'step_lr': 1000,
        'regularity': {'type': 'iid'}
    },
    'niter': 30,
    'dim_input': 1,
    'epochs': 10,
    'n_workers': 5,
    'nb_classes': 3
}

weights_after_training = {
    'name': 'weights-after-training',
    'model': 'SimpleResNet',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'lr': 0.01,
        'step_lr': 1000
    },
    'epochs': 10,
    'n_workers': 5
}

perf_weights_regularity = {
    'name': 'perf-weights-regularity-dataset',
    'model': 'SimpleResNet',
    'dataset': 'MNIST',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'scaling': 1.,
        'regularity':
            {
                'type': 'fbm',
                'value': 0.5
            },
        'lr': 0.001,
        'step_lr': 5
    },
    'epochs': 10,
    'n_workers': 5
}

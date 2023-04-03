
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
    'epochs': 300,
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

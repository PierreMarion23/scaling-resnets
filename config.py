
scaling_initialization_exp = {
    'name': 'scaling_initialization_exp',
    'model': 'FullResnet',
    'model-config': {
        'width': 40,
        'depth': 10,
        'scaling_beta': 0.5,
        'activation': 'ReLU',
        'regularity': {}
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
        'scaling_beta': 0.5,
        'activation': 'ReLU',
        'regularity': {'type': 'iid'}
    },
    'niter': 10 ** 4,
    'dim_input': 64,
    'nb_classes': 1,
}


perf_weights_regularity = {
    'name': 'perf-weights-regularity-dataset',
    'dataset': ['MNIST'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'SimpleResNet',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'scaling_beta': 0.1,
        'regularity':
            {
                'type': 'fbm',
                'value': 0.5
            },
        'lr': 0.0001,
        'step_lr': 5
    },
    'epochs': 10,
    'n_workers': 5
}

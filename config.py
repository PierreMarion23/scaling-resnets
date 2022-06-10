perf_weights_regularity = {
    'name': 'perf-weights-regularity',
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

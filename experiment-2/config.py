perf_weights_regularity = {
    'name': 'perf-weights-regularity',
    'dataset': ['MNIST'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,  # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': False,
        'train_final': False,
        'scaling': 'beta',
        'scaling_beta': 0.1,
        'lr': 0.0001,
        'regularity': 0.5,
        'step_lr': 10
    },
    'epochs': 10,
    'n_workers': 8
}

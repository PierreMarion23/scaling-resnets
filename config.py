standard = {
    'dataset': 'MNIST',                    # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 30
}

debug = {
    'dataset': 'MNIST',                    # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 2
}
standard_config = {
    'dataset': 'MNIST',                    # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'batch_norm': False,
        'skip_connection': True,
        'train_init': False,
        'train_final': False
    },
    'epochs': 30
}
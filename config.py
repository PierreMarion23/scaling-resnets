standard = {
    'name': 'dataset-standard',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 30
}

scaling_speedup = {
    'name': 'dataset-scaling-speedup-iid',
    'dataset': ['FashionMNIST'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 100,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,  # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': False,
        'train_final': False,
        'scaling': 'beta',
        'scaling_beta': 0.1,
        'lr': 0.0001
    },
    'epochs': 30
}

tanh = {
    'name': 'dataset-tanh',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 200,
        'activation': 'Tanh',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 20
}

low_noise = {
    'name': 'dataset-low-noise',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 0.1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 0.1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 30
}

high_noise = {
    'name': 'dataset-high-noise',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 10,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 10,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 30
}

batch_norm = {
    'name': 'dataset-batch-norm',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': True,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 30
}

no_train_init_final = {
    'name': 'dataset-no-train-init-final',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': False,
        'train_final': False,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 30
}

high_depth = {
    'name': 'dataset-high-depth',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 30
}

low_width = {
    'name': 'dataset-low-width',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 6,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 30
}

no_skip_connection = {
    'name': 'dataset-no-skip',
    'dataset': ['MNIST'],                    # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 4,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': False,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 30
}

debug = {
    'name': 'debug',
    'dataset': ['MNIST'],                    # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'beta',
        'scaling_beta': 1.,
        'lr': 0.1
    },
    'epochs': 2
}

debug_cnn = {
    'name': 'debug-cnn',
    'dataset': ['SVHN'],                    # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'ConvResNet',
    'full-logs': False,
    'model-config': {
        'channels': 10,
        'depth': 20,
    },
    'epochs': 2
}

no_scaling = {
    'name': 'dataset-sqrt-scaling',
    'dataset': ['MNIST'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'none',
        'lr': 0.1
    },
    'epochs': 10
}

rezero = {
    'name': 'dataset-no-scaling',
    'dataset': ['CIFAR10'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': 'FCResNet',
    'model-config': {
        'width': 64,
        'depth': 64,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True,
        'scaling': 'none',
        'lr': 0.1
    },
    'epochs': 100
}

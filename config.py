standard = {
    'name': 'dataset-standard',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 30
}

tanh = {
    'name': 'dataset-tanh',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'Tanh',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 30
}

low_noise = {
    'name': 'dataset-low-noise',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 0.1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 0.1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 30
}

high_noise = {
    'name': 'dataset-high-noise',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 10,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 10,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 30
}

batch_norm = {
    'name': 'dataset-batch-norm',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': True,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 30
}

no_train_init_final = {
    'name': 'dataset-no-train-init-final',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': False,
        'train_final': False
    },
    'epochs': 30
}

high_depth = {
    'name': 'dataset-high-depth',
    'dataset': ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'],  # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 1000,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 30
}

no_skip_connection = {
    'name': 'dataset-no-skip',
    'dataset': 'MNIST',                    # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 4,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': False,
        'train_init': True,
        'train_final': True
    },
    'epochs': 30
}

debug = {
    'name': 'debug',
    'dataset': 'MNIST',                    # one of {'MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN'}
    'model': {
        'width': 30,
        'depth': 200,
        'activation': 'ReLU',              # 'ReLU' or 'Tanh'
        'init_final_initialization_noise': 1,         # Initialization noise as a fraction of the default noise
        'layers_initialization_noise': 1,
        'batch_norm': False,
        'skip_connection': True,
        'train_init': True,
        'train_final': True
    },
    'epochs': 2
}
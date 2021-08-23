import os
import numpy as np
import torch
import torch.nn as nn

from generate_dataset import generate_data_ODE, generate_mnist_resnet
from scaling_model import FinalResNet
from utils import (
    experiment_results,
    plot_specific_entries,
    OPTIMIZERS,
    save_loss_vs_depth,
    save_weights,
    save_losses_during_training,
    save_gradients_during_training,
    gradient_magnitude,
    save_gradients
)

from regression import make_regression


def train_final_net(
    net,
    trainloader,
    testloader,
    criterion,
    num_epochs,
    optimizer,
    path,
    batch_size,
    epsilon,
    is_last_depth,
    sample_entries,
    save
):

    nb_batches = int(trainloader.shape[0] / batch_size)
    trainloader = trainloader.reshape(nb_batches, batch_size, trainloader.shape[-1])

    # best_epoch_loss = np.inf
    # bad_epoch = 0
    epoch = 0
    train_losses = []
    test_losses = []
    gradients = []

    while epoch < num_epochs: # and bad_epoch < 5:
        trainloader = trainloader[torch.randperm(trainloader.size()[0])]

        epoch_loss = 0.0

        net.train()
        if save:
            save_weights(net, path, epoch)

        for i in range(trainloader.shape[0]):
            ###### get data and forward pass ######
            data = torch.Tensor(trainloader[i])
            inputs, target = data[:, : net.dim], data[:, net.dim :]
            optimizer.zero_grad()
            outputs = net(inputs)
            if criterion.__str__() == "CrossEntropyLoss()":
                target = target.squeeze().long()
            loss = criterion(outputs, target)
            epoch_loss += loss.data.numpy() / nb_batches

            ###### backprop ######
            loss.backward()
            gradients.append(gradient_magnitude(net))
            if save:
                save_gradients(net, path, epoch, i)
            optimizer.step()

        train_losses.append(epoch_loss)
        net.eval()
        inputs, target = testloader[:, : net.dim], testloader[:, net.dim :]
        outputs = net(inputs)
        test_losses.append(criterion(outputs, target).data.numpy())
        # if epoch_loss < 0.99 * best_epoch_loss:
        #     best_epoch_loss = epoch_loss
        #     bad_epoch = 0
        # else:
        #     bad_epoch += 1

        if epoch % 5 == 4:
            print(f"    Epoch: {epoch + 1}/{num_epochs}.")
            print(f"    Train MSE: {np.round(epoch_loss, 5)}.")

        if is_last_depth:
            plot_specific_entries(
                net, path, save=True, entries=sample_entries, epoch=epoch
            )

        # if epoch_loss < epsilon:
        #     print(f"    Converged in {epoch + 1} epochs.")
        #     return train_losses

        epoch += 1

    print(f"    Did not converge. Remaining loss: {np.round(epoch_loss, 5)}.")
    return train_losses, test_losses, gradients


def random_entry(dim):
    return (np.random.choice(dim), np.random.choice(dim))


def run_experiment(
    delta_type="none",  # 'none', shared', 'multi'
    initial_sd=1.0e-04,
    init_method="xavier",
    activation="relu",  # 'relu', 'tanh'
    dim=10,
    dataset="ODE",
    optimizer_name="sgd",
    num_epochs=200,
    epsilon=1.0e-02,
    train_size=1024,
    test_size=256,
    batch_size=50,
    lr=5.0e-03,
    path="./scaling/",
    save=True,
    min_depth=3,
    max_depth=1000,
    base=1.2,  # need: base**n < max_depth
    seed=None,
    device='cpu'
):
    """Generic function to run a single experiment."""

    if seed is not None:
        torch.manual_seed(seed)

    final_train_losses = {}
    final_test_losses = {}

    if dataset == "ODE":
        train_set, test_set = generate_data_ODE(
            d=dim,
            f=np.sin,
            g=np.cos,
            noise=0.0,
            train_size=train_size,
            test_size=test_size,
            n_steps=100,
        )

    elif dataset == "mnist":
        dim = 25
        train_set, test_set = generate_mnist_resnet(batch_size=batch_size)

    else:
        raise NotImplementedError(
            f"Unknown dataset {dataset}. Allowed values: 'ODE' and 'mnist'."
        )

    # save dataset
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(train_set, path + "train_set.p")
    torch.save(test_set, path + "test_set.p")

    def get_depths(min_depth: int, max_depth: int, base: float = 1.25) -> list:
        x = 1 + int(np.log(max_depth) / np.log(base))
        depths = [int(np.floor(base ** n)) for n in range(x)]
        return np.unique([d for d in depths if d >= min_depth])

    depths = get_depths(min_depth=min_depth, max_depth=max_depth, base=base)

    for idx, depth in enumerate(depths, 1):
        path_with_depth = path + f"depth_{depth:03d}/"
        if not os.path.exists(path_with_depth):
            os.makedirs(path_with_depth)

        net = FinalResNet(
            dim=dim,
            num_layers=depth,
            delta_type=delta_type,  # 'none', 'shared', 'multi'
            initial_sd=initial_sd,
            activation=activation,  # 'relu', 'tanh'
        ).to(torch.device(device))

        criterion = nn.MSELoss()

        net.init_values(method=init_method)

        parameters = net.A + net.b
        if net.delta_type == "shared":
            parameters += [net.delta]
        elif net.delta_type == "multi":
            parameters += net.delta
        optimizer = OPTIMIZERS[optimizer_name](parameters, lr=lr * depth)

        eval_indices = np.random.randint(low=0, high=train_set.shape[0], size=(5,))
        train_samples = train_set[eval_indices, : net.dim]

        nb_batches = 5
        sample_entries = (
            [random_entry(net.dim) for _ in range(5)],
            [np.random.choice(net.dim) for _ in range(nb_batches)],
        )

        print("\n  Training at depth %d \n" % depth)

        train_losses, test_losses, gradients = train_final_net(
            net=net,
            trainloader=train_set.to(torch.device(device)),
            testloader=torch.Tensor(test_set).to(torch.device(device)),
            criterion=criterion,
            num_epochs=num_epochs,
            optimizer=optimizer,
            path=path_with_depth,
            batch_size=batch_size,
            epsilon=epsilon,
            is_last_depth=idx == len(depths),
            sample_entries=sample_entries,
            save=save
        )

        final_train_losses.update({depth: train_losses[-1]})
        final_test_losses.update({depth: test_losses[-1]})
        save_losses_during_training(train_losses, test_losses, path_with_depth)
        save_gradients_during_training(gradients, path_with_depth)
        
        experiment_results(
            net=net,
            path=path_with_depth,
            save=save,
            train_samples=train_samples,
            sample_entries=sample_entries,
        )

    save_loss_vs_depth(sorted(final_train_losses.items()),
                       sorted(final_test_losses.items()), 
                       path=path, epsilon=epsilon)

    if not os.path.exists(path + "regression/"):
        os.makedirs(path + "regression/")

    make_regression(net=net, path=path)

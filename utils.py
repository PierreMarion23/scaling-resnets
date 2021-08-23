import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from torch.optim import SGD, Adam, RMSprop


OPTIMIZERS = {"sgd": SGD, "adam": Adam, "rmsprop": RMSprop}
DATA_FUNCTIONS = {"linear": lambda x: x, "relu": lambda x: x * (x > 0), "tanh": np.tanh}


def custom_round(beta, d=1):
    # avoid having \beta = -0.0 in plots
    if np.round(beta, d) == 0:
        return 0.0
    else:
        return np.round(beta, d)


def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_weights(net, path, iteration):
    dir = os.path.join(path, "weights", str(iteration))
    if not os.path.exists(dir):
        os.makedirs(dir)

    torch.save(net.A, os.path.join(dir, "A.p"))
    torch.save(net.b, os.path.join(dir, "b.p"))
    torch.save(net.delta, os.path.join(dir, "C.p"))


def save_gradients(net, path, iteration, batch_number):
    dir = os.path.join(path, "weights", str(iteration), str(batch_number))
    if not os.path.exists(dir):
        os.makedirs(dir)

    torch.save([a.grad for a in net.A], os.path.join(dir, "gradA.p"))
    torch.save([b.grad for b in net.b], os.path.join(dir, "gradb.p"))
    if type(net.delta) == list:
        torch.save([d.grad for d in net.delta], os.path.join(dir, "gradC.p"))
    else:
        torch.save(net.delta.grad, os.path.join(dir, "gradC.p"))


def experiment_results(net, path, save, train_samples, sample_entries):
    if not os.path.exists(path + "weights/"):
        os.makedirs(path + "weights/")

    a_norms = [np.linalg.norm(x.data.numpy(), ord=2) for x in net.A]
    print(f"    Spectral norm of A: {np.max(a_norms)}.")
    plt.xlabel(r"Layer $k$")
    plt.plot(
        np.arange(len(net.A)),
        a_norms,
        "bo",
        markersize=0.5,
        label="Spectral norms $||A_k||_2$",
    )
    plt.legend(loc="upper left")
    if save:
        plt.savefig(path + "weights/A")
        torch.save(net.A, path + "weights/A.p")
    plt.clf()

    b_norms = [np.linalg.norm(x.data.numpy(), ord=2) for x in net.b]
    print(f"    Spectral norm of b: {np.max(b_norms)}.")
    plt.xlabel(r"Layer $k$")
    plt.plot(
        np.arange(len(net.b)),
        b_norms,
        "bo",
        markersize=0.5,
        label="Euclidean norm $||b_k||_2$",
    )
    plt.legend(loc="upper left")
    if save:
        plt.savefig(path + "weights/b")
        torch.save(net.b, path + "weights/b.p")
    plt.clf()

    if net.delta_type in ["multi", "shared"]:
        deltas = net.delta
        if not isinstance(net.delta, list):
            deltas = [net.delta] * net.num_layers

        C_norms = [np.linalg.norm(x.data.numpy(), ord=2) for x in deltas]
        print(f"    Spectral norm of C: {np.max(C_norms)}.")
        plt.title(r"Norm of $C_\ell$")
        plt.xlabel(r"Layer $\ell$")
        plt.ylabel(r"$||C_\ell||$")
        plt.plot(np.arange(len(deltas)), C_norms, "bo", markersize=0.5)
        if save:
            plt.savefig(path + "weights/C")
            torch.save(net.delta, path + "weights/C.p")
        plt.clf()

        delta_A_norms = [
            np.linalg.norm(c.data.numpy() * a.data.numpy(), ord=2)
            for c, a in zip(deltas, net.A)
        ]
        print(f"    Spectral norm of delta*A: {np.max(delta_A_norms)}.")
        plt.title(r"Norm $\delta_\ell A_\ell$")
        plt.xlabel(r"Layer $\ell$")
        plt.ylabel(r"$||\delta_\ell A_\ell||$")
        plt.plot(np.arange(len(deltas)), delta_A_norms, "bo", markersize=0.5)
        if save:
            plt.savefig(path + "weights/delta_times_A")
        plt.clf()

    plot_specific_entries(net, path, save=False, entries=sample_entries, epoch=None)

    plot_hidden_states(net, train_samples, save, path, True)


def plot_specific_entries(net, path, entries, save, epoch=None):
    if not os.path.exists(path + "paths/entries"):
        os.makedirs(path + "paths/entries")

    N = 5

    if net.delta_type == "matrix":
        A_entries, b_entries, C_entries = entries
    else:
        A_entries, b_entries = entries

    A = [[a.detach().numpy()[a_e] for a in net.A] for a_e in A_entries]
    b = [[b.detach().numpy()[b_e] for b in net.b] for b_e in b_entries]

    if net.delta_type == "matrix":
        C = [[c.detach().numpy()[c_e] for c in net.delta] for c_e in C_entries]

    for i in range(N):
        if not epoch:
            plt.title(f"Path A{A_entries[i]}, i: {i+1}/{N}")
            plt.plot(np.arange(net.num_layers), np.array(A[i]), "ko", markersize=0.5)
            plt.savefig(path + f"paths/A_path_{i}.png")
            plt.clf()
        elif save:
            np.save(
                path + f"paths/entries/A_{A_entries[i]}_epoch_{epoch}", np.array(A[i])
            )

        if not epoch:
            plt.title(f"Path b{b_entries[i]}, i: {i+1}/{N}")
            plt.plot(np.arange(net.num_layers), np.array(b[i]), "ko", markersize=0.5)
            plt.savefig(path + f"paths/b_path_{i}.png")
            plt.clf()
        elif save:
            np.save(
                path + f"paths/entries/b_{b_entries[i]}_epoch_{epoch}", np.array(b[i])
            )

        if net.delta_type == "matrix":
            if not epoch:
                plt.title(f"Path C{C_entries[i]}, i: {i+1}/{N}")
                plt.plot(
                    np.arange(net.num_layers), np.array(C[i]), "ko", markersize=0.5
                )
                plt.savefig(path + f"paths/C_path_{i}.png")
                plt.clf()
            elif save:
                np.save(
                    path + f"paths/entries/C_{C_entries[i]}_epoch_{epoch}",
                    np.array(C[i]),
                )


def plot_hidden_states(net, train_samples, save, path, is_trained):

    x = torch.Tensor(train_samples)
    hidden_states_list = [x]
    for l in range(net.num_layers):
        x = x + net.update_rule(x, l)
        hidden_states_list.append(x)

    # shape: (5, net.num_layers, net.dim)
    hidden_states = torch.stack(hidden_states_list).transpose(1, 0)
    hidden_diffs = hidden_states[:, 1:] - hidden_states[:, :-1]
    # shape: (5, net.num_layers)
    hidden_norms = torch.norm(hidden_states, dim=2)
    norms_diffs = torch.norm(hidden_diffs, dim=-1)

    colors = ["b--", "g--", "r--", "m--", "y--"]
    suff_path = "_" + ("after_train" if is_trained else "before_train")
    suff_title = " after training" if is_trained else " before training"

    plt.title(r"In-sample $||h_\ell||$" + suff_title)
    plt.xlabel(r"Layer $\ell$")
    plt.ylabel(r"$||h_\ell||$")
    t = np.arange(net.num_layers + 1)
    for i in range(5):
        plt.plot(t, hidden_norms[i].data.numpy(), colors[i])
    if save:
        plt.savefig(path + "in_sample_norms_hidden_states" + suff_path)
    plt.clf()

    plt.title(r"In-sample $||h_{\ell+1}-h_{\ell}||$" + suff_title)
    plt.xlabel(r"Layer $\ell$")
    plt.ylabel(r"$||h_{\ell+1}-h_{\ell}||$")
    t = np.arange(net.num_layers)
    for i in range(5):
        plt.plot(t, norms_diffs[i].data.numpy(), colors[i])
    if save:
        plt.savefig(path + "in_sample_norms_hidden_diffs" + suff_path)
    plt.clf()

    x = torch.Tensor(np.random.uniform(-1, 1, (5, net.dim)))
    x /= torch.norm(x, dim=1, keepdim=True)
    hidden_states_list = [x]
    for l in range(net.num_layers):
        x = x + net.update_rule(x, l)
        hidden_states_list.append(x)

    # shape: (5, net.num_layers, net.dim)
    hidden_states = torch.stack(hidden_states_list).transpose(1, 0)
    hidden_diffs = hidden_states[:, 1:] - hidden_states[:, :-1]
    # shape: (5, net.num_layers)
    hidden_norms = torch.norm(hidden_states, dim=2)
    norms_diffs = torch.norm(hidden_diffs, dim=-1)

    plt.title(r"Out-of-sample $||h_\ell||$" + suff_title)
    plt.xlabel(r"Layer $\ell$")
    plt.ylabel(r"$||h_\ell||$")
    t = np.arange(net.num_layers + 1)
    for i in range(5):
        plt.plot(t, hidden_norms[i].data.numpy(), colors[i])
    if save:
        plt.savefig(path + "out_of_sample_norms_hidden_states" + suff_path)
    plt.clf()

    plt.title(r"Out-of-sample $||h_{\ell+1}-h_{\ell}||$" + suff_title)
    plt.xlabel(r"Layer $\ell$")
    plt.ylabel(r"$||h_{\ell+1}-h_{\ell}||$")
    t = np.arange(net.num_layers)
    for i in range(5):
        plt.plot(t, norms_diffs[i].data.numpy(), colors[i])
    if save:
        plt.savefig(path + "out_of_sample_norms_hidden_diffs" + suff_path)
    plt.clf()


def save_loss_vs_depth(dic_training_losses, dic_test_loss, path, epsilon):
    training_depths, training_losses = zip(*dic_training_losses)
    test_depths, test_losses = zip(*dic_test_loss)
    xmin = int(0.9 * min(training_depths))
    xmax = int(1.1 * max(training_depths))

    fig, ax = plt.subplots(tight_layout=True)
    plt.xlabel(r"Depth")
    ax.set_xlim(xmin, xmax)
    plt.plot(training_depths, training_losses, "k--", lw=1, label="Final training loss")
    plt.plot(test_depths, test_losses, "r", lw=1, label="Final test loss")
    if epsilon:
        plt.fill_between(x=[xmin, xmax], y1=epsilon, color="#CDE7F0")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.savefig(path + "loss_vs_depth.png")
    save_pickle(path + "training_loss_vs_depth.p", dic_training_losses)
    save_pickle(path + "test_loss_vs_depth.p", dic_test_loss)
    plt.clf()


def save_losses_during_training(training_losses, test_losses, path):
    fig, ax = plt.subplots(tight_layout=True)
    plt.xlabel(r"Epoch")
    plt.plot(range(len(training_losses)), training_losses, "k--", lw=1, label="Training loss")
    plt.plot(range(len(test_losses)), test_losses, "r", lw=1, label="Test loss")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.savefig(path + "losses.png")
    save_pickle(path + "training_loss.p", training_losses)
    save_pickle(path + "test_loss.p", test_losses)
    plt.clf()


def save_gradients_during_training(gradients, path):
    fig, ax = plt.subplots(tight_layout=True)
    plt.xlabel(r"Step")
    plt.plot(range(len(gradients)), gradients, "k--", lw=1, label="Gradient magnitude")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.savefig(path + "gradients.png")
    save_pickle(path + "gradients.p", gradients)
    plt.clf()


def save_single_delta(delta, path, L):
    plt.title(r"Value of $\delta$ across $L$ = " + str(L) + " layers")
    plt.xlabel(r"Layer $\ell$")
    plt.ylabel(r"$\delta_{\ell}$")
    l = np.arange(L)
    plt.plot(l, delta)
    plt.savefig(path + "delta.png")
    plt.clf()


def gradient_magnitude(model: torch.nn.Module) -> torch.Tensor:
    return sum([torch.norm(x.grad) for x in model.A]).detach().numpy() + sum([torch.norm(x.grad) for x in model.b]).detach().numpy()

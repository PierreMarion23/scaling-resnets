import numpy as np
import pytorch_lightning as pl
import torch
import os
import pickle
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

def get_prediction(data, model: pl.LightningModule, device):
    model.eval() # Deactivates gradient graph construction during eval.
    data = data.to(device)
    model.to(device)
    probabilities = torch.softmax(model(data), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities

def pred(test_dl, model, device):
    targets, predictions = [], []
    for batch in iter(test_dl):
        data, target = batch
        targets.extend(target)
        model.eval()
        data = data.to(device)
        model.to(device)
        predictions.extend(model(data).cpu().detach().numpy())
    return targets, predictions

def get_true_targets_predictions(test_dl, model, device):
    true_targets, predictions = [], []
    for batch in iter(test_dl):
        data, target = batch
        true_targets.extend(target)
        prediction, _ = get_prediction(data, model, device)
        predictions.extend(prediction.cpu())
    return true_targets, predictions


def get_eval_loss(test_dl, model, device):
    model.eval()
    loss = 0
    for n, batch in enumerate(test_dl):
        data, target = batch
        logits = model(data.to(device))
        loss += model.loss(logits, target.to(device))
    return loss / n


def generate_fbm(N, H):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method

    args:
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k, H: 0.5 * (
                np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(
            k + 1) ** (2 * H))
    g = [gamma(k, H) for k in range(0, N)]
    r = g + [0] + g[::-1][0:N - 1]

    # Step 1 (eigenvalues)
    j = np.arange(0, 2 * N)
    k = 2 * N - 1
    lk = np.fft.fft(
        r * np.exp(2 * np.pi * complex(0, 1) * k * j * (1 / (2 * N))))[::-1]

    # Step 2 (get random variables)
    Vj = np.zeros((2 * N, 2), dtype=complex)
    Vj[0, 0] = np.random.standard_normal()
    Vj[N, 0] = np.random.standard_normal()

    for i in range(1, N):
        Vj1 = np.random.standard_normal()
        Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1
        Vj[i][1] = Vj2
        Vj[2 * N - i][0] = Vj1
        Vj[2 * N - i][1] = Vj2

    # Step 3 (compute Z)
    wk = np.zeros(2 * N, dtype=complex)
    wk[0] = np.sqrt((lk[0] / (2 * N))) * Vj[0][0]
    wk[1:N] = np.sqrt(lk[1:N] / (4 * N)) * (
                (Vj[1:N].T[0]) + (complex(0, 1) * Vj[1:N].T[1]))
    wk[N] = np.sqrt((lk[0] / (2 * N))) * Vj[N][0]
    wk[N + 1:2 * N] = np.sqrt(lk[N + 1:2 * N] / (4 * N)) * (
                np.flip(Vj[1:N].T[0]) - (
                    complex(0, 1) * np.flip(Vj[1:N].T[1])))

    Z = np.fft.fft(wk)
    fGn = Z[0:N]
    fBm = np.cumsum(fGn) * (N ** (-H))
    path = np.array([0] + list(fBm))
    return path, fGn

def generate_volterra(alpha, dim: int, As: list[np.array], bs: list[np.array]):
    """Simulate a volterra process

    :param alpha: K(t)=t^(alpha-1)
    :param dim: dimension of the process
    :param As: a(x)=As[0]+x_1As[1]+...+x_dAs[d]
    :param bs: b(x)=bs[0]+x_1bs[1]+...+x_dbs[d]
    return
    """
    


def rbf_kernel(x1, x2, bandwidth):
    return np.exp(-1 * ((x1-x2) ** 2) / (2*bandwidth**2))


def cov_matrix_for_rbf_kernel(depth, bandwidth):
    xs = np.linspace(0, 1, depth + 1)
    return [[rbf_kernel(x1, x2, bandwidth) for x2 in xs] for x1 in xs]


def rbf_kernel_multivariate(x1, x2, bandwidth, cov):
    width = len(x1)
    kernel = np.zeros((width, width))
    for i in range(width):
        for j in range(width):
            kernel[i, j] = np.exp(
                                -1 * (cov[i, i]*x1[i]*x1[i] + cov[j, j]*x2[j]*x2[j]-\
                                2*cov[i, j]*x1[i]*x2[j])/ (2*bandwidth**2))
    return kernel

def cov_matrix_for_rbf_with_cov(width, depth, bandwidth, cov):
    xs = np.linspace(0, 1, depth+1)
    res = [[0]*(depth+1) for _ in range(depth+1)]
    for i, x1 in enumerate(xs):
        for j, x2 in enumerate(xs):
            x1_multi = np.full(width*width, x1)
            x2_multi = np.full(width*width, x2)
            res[i][j] = rbf_kernel_multivariate(x1_multi, x2_multi, bandwidth, cov)
    filepath = "pickles/scaling_initialization"
    os.makedirs(filepath, exist_ok=True)
    cov_matrix = np.concatenate([np.concatenate(row, axis = 1) for row in res])
    print(cov_matrix)
    with open(os.path.join(filepath, f"smooth_cov_matrix_depth{depth}.pkl"), "wb") as f:
        pickle.dump(cov_matrix, f)
        print(f"cov_matrix of depth {depth} saved")
    return cov_matrix


def create_cov_matrix(size, seed) -> np.array:
    """
    Create a semi-definite positive matrix with diagonal 1
    """
    rng = np.random.default_rng(seed=seed)
    A = rng.random((size*size, size*size))
    return np.corrcoef(A)

def generate_heston_paths(T, hurst, theta=0.02, lam=0.3, v_0=0.02, mu=0.3,
                          steps=1000):
    def K(t):
        return t ** (hurst-0.5)
    dt = T/steps
    Vs = np.zeros(steps+1)
    Ws = np.zeros(steps+1)
    Vs[0]=v_0
    for t in range(1, steps+1):
        increment_W = np.sqrt(dt) * np.random.standard_normal()
        Ws[t-1] = increment_W
        Ks = np.array([K(k*dt) for k in range(t, 0, -1)])
        V_plus = np.maximum(0, Vs)
        V = v_0 + dt * np.dot(Ks, (theta-lam*V_plus[:t]))\
                + mu * np.dot(Ks, np.sqrt(V_plus[:t])*Ws[:t])
        
        Vs[t] = V
    
    return Vs

def max_norm(l1: List[torch.Tensor], 
             l2: Optional[List[torch.Tensor]] = None) -> float:
    diff = 0
    if not l2 is None:
        for v1, v2 in zip(l1, l2):
            diff = max(diff, torch.norm(v1-v2).item())
    else:
        for v1 in l1:
            diff = max(diff, torch.norm(v1).item())
    return diff

def max_norm_prop(l1:list, l2:list):
    diff = 0
    for v1, v2 in zip(l1, l2):
        diff = max(diff, torch.norm(v1-v2).item()/torch.norm(v2).item())
    return diff

def mean_norm(l1:list, l2:list):
    diff = 0
    for v1, v2 in zip(l1, l2):
        diff = diff + (torch.norm(v1-v2).item() ** 2)
    diff = diff/len(l1)
    return diff

if __name__ == "__main__":
    res = generate_heston_paths(1, 0.001, v_0=0.5)
    plt.plot(np.linspace(0, 1, 1001), res)
    plt.savefig("figures/volterra.png")
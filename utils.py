import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.gaussian_process.kernels import RBF

def get_prediction(data, model: pl.LightningModule, device):
    model.eval() # Deactivates gradient graph construction during eval.
    data = data.to(device)
    model.to(device)
    probabilities = torch.softmax(model(data), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities


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


def rbf_kernel(x1, x2, bandwidth):
    return np.exp(-1 * ((x1-x2) ** 2) / (2*bandwidth**2))


def cov_matrix_for_rbf_kernel(depth, bandwidth):
    xs = np.linspace(0, 1, depth + 1)
    return [[rbf_kernel(x1, x2, bandwidth) for x2 in xs] for x1 in xs]


def rbf_kernel_multivariate(x1, x2, bandwidth, cov):
    return np.exp(-1 * (x1-x2).dot(cov).dot(x1-x2)/(2*bandwidth**2))
def cov_matrix_for_rgf_with_cov(width, depth, bandwidth, cov):
    xs = np.linspace(0, 1, depth+1)
    cov_res = [[0] * len(xs) for _ in xs]
    for i, x1 in enumerate(xs):
        for j, x2 in enumerate(xs):
            x1_multi = np.array([x1]*(width*width))
            x2_multi = np.array([x2]*(width*width))
            cov_res[i][j] = rbf_kernel_multivariate(x1_multi, x2_multi, bandwidth, cov)
    cov_new = np.concatenate([np.concatenate(row, axis = 1) for row in cov_res])
    return cov_new
def create_cov_matrix(size, seed):
    """
    Create a semi-definite positive matrix with diagonal 1
    """
    rng = np.random.default_rng(seed=seed)
    A = rng.random((size*size, size*size))
    return np.corrcoef(A)
import numpy as np
from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt

import models
import utils
import wandb


class PrintingCallback(Callback):
    def __init__(self, test_dl, device, full_logs):
        super().__init__()
        self.test_dl = test_dl
        self.device = device
        self.full_logs = full_logs

    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/5238#issuecomment-750351799
    def on_after_backward(self, trainer, pl_module):
        if not self.full_logs:
            return
        logging_depths = [0,
                          int(pl_module.depth / 4),
                          int(pl_module.depth / 2),
                          int(3 * pl_module.depth / 4),
                          pl_module.depth - 1]

        if trainer.global_step % 25 == 0:
            idx_depth = 0
            # We want information about gradients, so we cannot use state_dict(), but we use parameters().
            for param in pl_module.parameters():
                if param.requires_grad and param.grad is not None and param.shape == (pl_module.width, pl_module.width):
                    if idx_depth in logging_depths:
                        # Log gradient norms
                        pl_module.logger.experiment.log(
                            {"train/grad-norm/" + str(idx_depth): torch.norm(param.grad), "global_step": trainer.global_step})

                        # Log weight norms
                        pl_module.logger.experiment.log(
                            {"train/weight-norm/" + str(idx_depth): torch.norm(param), "global_step": trainer.global_step})
                    idx_depth += 1

            # Log scaling weights
            if pl_module.scaling == "rezero":
                scaling_weights = pl_module.resweight
                for i in range(pl_module.depth):
                    pl_module.logger.experiment.log(
                        {"train/scaling/" + str(i): torch.abs(scaling_weights[i]),
                         "global_step": trainer.global_step})

    def on_epoch_end(self, trainer, pl_module):
        # Log test accuracy
        true_targets, predictions = utils.get_true_targets_predictions(
            self.test_dl, pl_module, self.device)
        accuracy = np.mean(np.array(true_targets) == np.array(predictions))
        loss = utils.get_eval_loss(self.test_dl, pl_module, self.device)
        pl_module.logger.experiment.log(
            {"test/accuracy/true": accuracy, "global_step": trainer.global_step})
        pl_module.logger.experiment.log(
            {"test/loss/true": loss, "global_step": trainer.global_step})

        if not self.full_logs:
            return


        # Log test accuracy when the ODE is replaced by identity.
        denoised_model = pl_module.copy()
        new_weights = list(denoised_model.layers.children())
        new_weights2 = list(denoised_model.inner_weights.children())
        for k in range(pl_module.depth):
            new_weights[k].weight = torch.nn.Parameter(
                torch.zeros((pl_module.width, pl_module.width)))
            new_weights[k].bias = torch.nn.Parameter(
                torch.zeros((pl_module.width,)))
        true_targets, predictions = utils.get_true_targets_predictions(
            self.test_dl, denoised_model, self.device)
        accuracy = np.mean(np.array(true_targets) == np.array(predictions))
        pl_module.logger.experiment.log(
            {"test/accuracy/zeroed": accuracy, "global_step": trainer.global_step})

        # Log example of polynomial approximation for one weight
        layers = list(pl_module.layers.children())
        inner_weights = list(pl_module.inner_weights.children())
        weight_example = np.array(
            [layers[k].weight[4, 3].detach().numpy() for k in range(pl_module.depth)])
        weight_example2 = np.array(
            [inner_weights[k].weight[4, 3].detach().numpy() for k in range(pl_module.depth)])
        poly_example, _ = models.polyfit(weight_example, 3)
        poly_example2, _ = models.polyfit(weight_example2, 3)
        plt.plot(weight_example)
        plt.plot(poly_example)
        plt.plot(weight_example2)
        plt.plot(poly_example2)
        trainer.logger.experiment.log({
            "Example of polynomial approximation for a weight": plt,
            "global_step": trainer.global_step
            })

        for degree in [1, 2, 3, 5]:
            denoised_weights, denoised_bias, errors_weights, _ = models.denoise_weights_bias(
                layers, degree)
            denoised_weights2, denoised_bias2, errors_weights2, _ = models.denoise_weights_bias(
                inner_weights, degree)
            # Log error of polynomial approximation
            pl_module.logger.experiment.log({
                "train/poly-error/degree-" + str(degree): np.linalg.norm(errors_weights) + np.linalg.norm(errors_weights2),
                "global_step": trainer.global_step
                })
            # Log norm of polynomial approximation
            pl_module.logger.experiment.log({
                "train/poly-norm/degree-" + str(degree): np.linalg.norm(denoised_weights) + np.linalg.norm(denoised_weights2),
                "global_step": trainer.global_step
                })

            # Log accuracy of polynomial approximation
            for k in range(pl_module.depth):
                new_weights[k].weight = torch.nn.Parameter(
                    torch.Tensor(denoised_weights[k]))
                new_weights[k].bias = torch.nn.Parameter(
                    torch.Tensor(denoised_bias[k]))
                new_weights2[k].weight = torch.nn.Parameter(
                    torch.Tensor(denoised_weights2[k]))
                new_weights2[k].bias = torch.nn.Parameter(
                    torch.Tensor(denoised_bias2[k]))
            true_targets, predictions = utils.get_true_targets_predictions(
                self.test_dl, denoised_model, self.device)
            accuracy = np.mean(np.array(true_targets) == np.array(predictions))
            pl_module.logger.experiment.log({
                "test/accuracy/degree-" + str(degree): accuracy,
                "global_step": trainer.global_step
                })

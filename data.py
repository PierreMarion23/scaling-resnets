import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data


# This is to reshape MNIST (28, 28) tensors into (784, ) tensors.
# See https://discuss.pytorch.org/t/missing-reshape-in-torchvision/9452/7
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size
    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def mnist():
    mnist_transforms = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    train_ds = torchvision.datasets.MNIST(
        "mnist", train=True, download=True, transform=mnist_transforms)
    test_ds = torchvision.datasets.MNIST(
        "mnist", train=False, download=True, transform=mnist_transforms)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=3)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64)
    return train_dl, test_dl

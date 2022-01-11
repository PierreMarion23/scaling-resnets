import os

import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data


# This is useful to reshape 2d image tensors into a 1d-tensor.
# See https://discuss.pytorch.org/t/missing-reshape-in-torchvision/9452/7
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size
    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def load_dataset(name):
    transform_img_to_vect = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    try:
        dataset_class = getattr(torchvision.datasets, name)
    except AttributeError:
        raise ValueError('%s is not a torchvision dataset.' % name)
    train_arg = {'split': 'train'} if name == 'SVHN' else {'train': True}
    test_arg = {'split': 'test'} if name == 'SVHN' else {'train': False}
    train_ds = dataset_class(
        os.path.join("datasets", name), download=True, transform=transform_img_to_vect, **train_arg)
    test_ds = dataset_class(
        os.path.join("datasets", name), download=True, transform=transform_img_to_vect, **test_arg)
    # This freezes the seed for the dataloader.
    generator = torch.Generator()
    generator.manual_seed(0)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=0, generator=generator)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64)
    size = np.prod(np.array(train_ds.data[0].shape))
    nb_classes = len(np.unique(train_ds.labels)) if name == 'SVHN' else len(train_ds.classes)
    return train_dl, test_dl, size, nb_classes

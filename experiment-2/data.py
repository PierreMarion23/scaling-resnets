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


def load_dataset(name, vectorize):
    if vectorize:
        transform_img_to_vect = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    else:
        transform_img_to_vect = transforms.Compose([transforms.ToTensor()])
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
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64, pin_memory=True)
    if vectorize:
        first_coord = np.prod(np.array(train_ds.data[0].shape))
    elif len(train_ds.data[0].shape) == 2: # Black and white image, so only one channel.
        first_coord = 1
    else:
        first_coord = train_ds.data[0].shape[0]
    nb_classes = len(np.unique(train_ds.labels)) if name == 'SVHN' else len(train_ds.classes)
    return train_dl, test_dl, first_coord, nb_classes
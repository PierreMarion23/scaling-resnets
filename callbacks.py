import copy
import glob
from multiprocessing import Pool
import os
import pickle
import time
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch

class NormDiffCallback(Callback):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self._epoch = 0
    
    def on_train_epoch_end(self, trainer, pl_module):
        
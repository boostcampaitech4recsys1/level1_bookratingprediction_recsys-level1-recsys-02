import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from catboost import CatBoostClassifier

from ._models import rmse, RMSELoss

class DecisionTreeModel():

    def __init__(self, args, data) -> None:
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['filed_dims']

        self.classifier = CatBoostClassifier(
            iterations = 5,
            learning_rate = 0.1,
            #loss_function = 'CrossEntropy'
        )

        
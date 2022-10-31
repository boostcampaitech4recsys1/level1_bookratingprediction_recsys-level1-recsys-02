import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from ._models import rmse, RMSELoss


class DecisionTreeModel:
    def __init__(self, args, data) -> None:
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data["train_dataloader"]
        self.valid_dataloader = data["valid_dataloader"]
        self.field_dims = data["filed_dims"]

        self.x_train = self.train_dataloader[0]
        self.y_train = self.train_dataloader[1]

        self.x_valid = self.valid_dataloader[0]
        self.y_valid = self.valid_dataloader[1]

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.catboost_regressor = CatBoostRegressor(
            iterations=self.epochs, learning_rate=self.learning_rate, verbose=True
        )

    def train(self):
        self.catboost_regressor.fit(X=self.x_train, y=self.y_train)

    def predict_train(self):
        y_hat = self.catboost_regressor.predict(data=self.x_valid)
        return rmse(self.y_valid, y_hat)

    def predict(self, test_dataloader):
        return self.catboost_regressor.predict(data=test_dataloader)

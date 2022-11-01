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
        self.field_dims = data["field_dims"]

        self.iters = args.CATBOOST_ITERS
        self.learning_rate = args.LR
        self.depth = args.CATBOOST_DEPTH
        self.log_interval = 100

        self.device = args.DEVICE

        self.catboost_regressor = CatBoostRegressor(
            iterations=self.iters,
            learning_rate=self.learning_rate,
            verbose=True,
            depth=self.depth,
        )

    def train(self):
        for i, (x_train, y_train) in enumerate(tqdm(self.train_dataloader)):
            x_train = x_train.numpy()
            y_train = y_train.numpy()
            self.catboost_regressor.fit(X=x_train, y=y_train, verbose=False)
            # print(f"valid loss : {self.predict_train()}")
        print(f"valid loss : {self.predict_train()}")

    def predict_train(self):
        y = list()
        y_hat = list()
        for i, (x_valid, y_valid) in enumerate(self.valid_dataloader):
            x_valid = x_valid.numpy()
            y_valid = y_valid.numpy()
            y.extend(y_valid)
            temp = self.catboost_regressor.predict(data=x_valid)
            temp = np.around(temp)
            y_hat.extend(temp)
        return rmse(y, y_hat)

    def predict(self, test_dataloader):
        predicts = list()
        for i, x_test_batch in enumerate(tqdm(test_dataloader)):

            for x_test in x_test_batch:
                x_test = x_test.numpy()
                temp = self.catboost_regressor.predict(data=x_test)
                temp = np.around(temp)
                predicts.extend(temp)
        return predicts

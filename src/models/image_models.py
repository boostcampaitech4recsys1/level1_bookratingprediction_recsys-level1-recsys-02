import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from ._models import RMSELoss, FeaturesEmbedding, FactorizationMachine_v

from torchvision import models

class ResNet(nn.Module):
    def __init__(
            self,
            block,
            num_block,
            batch_size,
            init_weights=True
    ):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.batch_size = batch_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        if init_weights:
            self._initialize_weights()



    def _make_layer(
            self,
            block,
            out_channels,
            num_blocks,
            stride,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = list()

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = x.view(-1, 512)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class _ResNet_FM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, latent_dim, batch_size):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.resnet152 = ResNet(Block, [3, 8, 36, 3], batch_size)
        self.fm = FactorizationMachine_v(
                                         input_dim=(embed_dim * 2) + 512,
                                         latent_dim=latent_dim,
                                         )

    def forward(self, x):
        user_isbn_vector, img_vector = x[0], x[1]
        user_isbn_feature = self.embedding(user_isbn_vector)
        img_feature = self.resnet152(img_vector)

        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    img_feature
                                    ], dim=1)
        output = self.fm(feature_vector)

        return output.squeeze(1)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)

        return x

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * Block.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * Block.expansion),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != Block.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Block.expansion),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)

        return x

class CNN_Base(nn.Module):
    def __init__(
        self,
    ):
        super(CNN_Base, self).__init__()
        self.cnn_layer = nn.Sequential(
            # 32 32 3
            nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # 16 16 6
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # 8 8 12
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # 3 3 24
            nn.Conv2d(24, 40, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            # 1 1 40
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 40)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        for param in self.resnet.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        x = self.resnet.forward(x)
        return x


class _CNN_FM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, latent_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.cnn = CNN_Base()
        self.cnn = ResNet()
        self.fm = FactorizationMachine_v(
            input_dim=(embed_dim * len(field_dims)) + (2048),
            latent_dim=latent_dim,
        )

    def forward(self, x):
        context_vector, img_vector = x[0], x[1]
        context_feature = self.embedding(context_vector)
        img_feature = self.cnn(img_vector)
        feature_vector = torch.cat(
            [
                context_feature.view(
                    -1, context_feature.size(1) * context_feature.size(2)
                ),
                img_feature,
            ],
            dim=1,
        )
        output = self.fm(feature_vector)
        return output.squeeze(1)


class CNN_FM:
    def __init__(self, args, data):
        super().__init__()
        self.device = args.DEVICE
        self.model = _CNN_FM(
            field_dims=data["field_dims"],
            embed_dim=args.CNN_FM_EMBED_DIM,
            latent_dim=args.CNN_FM_LATENT_DIM,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.LR)
        self.train_data_loader = data["train_dataloader"]
        self.valid_data_loader = data["valid_dataloader"]
        self.criterion = RMSELoss()
        self.epochs = args.EPOCHS
        self.model_name = "image_model"
        if args.CNN_FM_LOAD_MODEL:
            self.model.load_state_dict(
                torch.load("./models/{}.pt".format(self.model_name))
            )

    def train(self):
        minimum_loss = 999999999
        loss_list = []
        tk0 = tqdm.tqdm(range(self.epochs), smoothing=0, mininterval=1.0)
        for epoch in tk0:
            self.model.train()
            total_loss = 0
            n = 0
            for i, data in enumerate(self.train_data_loader):
                if len(data) == 2:
                    fields, target = [data["context_vector"].to(self.device)], data[
                        "label"
                    ].to(self.device)
                else:
                    fields, target = [
                        data["context_vector"].to(self.device),
                        data["img_vector"].to(self.device),
                    ], data["label"].to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n += 1
            self.model.eval()
            val_total_loss = 0
            val_n = 0
            for i, data in enumerate(self.valid_data_loader):
                if len(data) == 2:
                    fields, target = [data["context_vector"].to(self.device)], data[
                        "label"
                    ].to(self.device)
                else:
                    fields, target = [
                        data["context_vector"].to(self.device),
                        data["img_vector"].to(self.device),
                    ], data["label"].to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                # self.model.zero_grad()
                val_total_loss += loss.item()
                val_n += 1
            if minimum_loss > (val_total_loss / val_n):
                minimum_loss = val_total_loss / val_n
                if not os.path.exists("./models"):
                    os.makedirs("./models")
                torch.save(
                    self.model.state_dict(), "./models/{}.pt".format(self.model_name)
                )
                loss_list.append(
                    [epoch, total_loss / n, val_total_loss / val_n, "Model saved"]
                )
            else:
                loss_list.append(
                    [epoch, total_loss / n, val_total_loss / val_n, "None"]
                )
            tk0.set_postfix(
                train_loss=total_loss / n, valid_loss=val_total_loss / val_n
            )
        for epoch in tqdm.tqdm(range(self.epochs), smoothing=0, mininterval=1.0):
            self.model.train()
            total_loss = 0
            n = 0
            for i, data in enumerate(self.valid_data_loader):
                if len(data) == 2:
                    fields, target = [data["context_vector"].to(self.device)], data[
                        "label"
                    ].to(self.device)
                else:
                    fields, target = [
                        data["context_vector"].to(self.device),
                        data["img_vector"].to(self.device),
                    ], data["label"].to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n += 1

    def predict(self, test_data_loader):
        self.model.eval()
        self.model.load_state_dict(torch.load("./models/{}.pt".format(self.model_name)))
        targets, predicts = list(), list()
        with torch.no_grad():
            for data in test_data_loader:
                if len(data) == 2:
                    fields, target = [data["context_vector"].to(self.device)], data[
                        "label"
                    ].to(self.device)
                else:
                    fields, target = [
                        data["context_vector"].to(self.device),
                        data["img_vector"].to(self.device),
                    ], data["label"].to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return predicts

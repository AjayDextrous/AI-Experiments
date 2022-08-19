import numpy
import pytorch_lightning as pl
import torch
import torch.nn as nn


class LinearClassifier(pl.LightningModule):
    def __init__(self, hparams, train_dataset, val_dataset, test_dataset):
        super().__init__()

        def initialize_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
            elif type(m) == nn.Conv2d:
                torch.nn.init.xavier_normal_(m.weight)

        self.save_hyperparameters(hparams)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        in_features = hparams.get('input_features', 1000)
        out_features = hparams.get('output_features', 10)
        act_fn = self.get_activation_function(hparams['activation_function'])

        layers = []
        layers.extend([
            nn.Linear(in_features, 1000),
            act_fn,
            nn.Linear(1000, 1000),
            act_fn,
            nn.Linear(1000, 100),
            act_fn,
            nn.Linear(100, out_features),
            nn.Softmax()
        ])

        for layer in layers:
            layer.apply(initialize_weights)
        self.model = nn.Sequential(*layers)

        pl.Trainer()

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def get_activation_function(func_str):
        switch = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
        }
        return switch.get(func_str, nn.LeakyReLU())

    def general_step(self, batch):
        images, targets = batch
        flattened_images = images.view(images.shape[0], -1)
        out = self.forward(flattened_images)
        loss = nn.functional.cross_entropy(out, targets)

        return loss

    def training_step(self, batch):
        loss = self.general_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        self.log("test_loss", loss)
        return loss

    # implement _end functions

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle =True, batch_size=self.hparams['batch_size'], num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle =True, batch_size=self.hparams['batch_size'], num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle =True, batch_size=self.hparams['batch_size'], num_workers=4)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr = self.hparams['lr'])
        return optim
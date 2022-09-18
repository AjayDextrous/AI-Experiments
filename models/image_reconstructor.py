import itertools
import numpy
import pytorch_lightning as pl
import torch
import torch.nn as nn


def get_accuracy(out, targets):
    batch_size = out.size()[0]
    correct_n = 0
    for result in range(batch_size):
        if torch.argmax(out[result]) == torch.argmax(targets[result]):
            correct_n += 1
    return correct_n / batch_size


class ConvolutionalReconstructor(pl.LightningModule):
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

        in_channels = hparams.get('input_channels', 1)
        kernel_size = hparams.get('kernel_size', 3)
        padding = hparams.get('padding', 1)
        act_fn = self.get_activation_function(hparams['activation_function'])

        encoder = []
        encoder.extend([
            nn.Conv2d(in_channels, 8, kernel_size=kernel_size, padding=padding),    # 28x28x1  -> 28x28x8
            nn.BatchNorm2d(8),
            act_fn,
            nn.MaxPool2d(2, stride=2),                                              # 28x28x8  -> 14x14x8
            nn.Conv2d(8, 16, kernel_size=kernel_size, padding=padding),             # 14x14x8  -> 14x14x16
            nn.BatchNorm2d(16),
            act_fn,
            nn.MaxPool2d(2, stride=2),                                              # 14x14x16 -> 7x7x16
            nn.Conv2d(16, 64, kernel_size=kernel_size, padding=padding),            # 7x7x16   -> 7x7x64
            nn.BatchNorm2d(64),
            act_fn,
            nn.MaxPool2d(3, stride=3, padding=1),                                   # 7x7x64   -> 3x3x64 = 576
        ])

        decoder = []
        decoder.extend([
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            act_fn,
            nn.UpsamplingNearest2d(size=7),
            nn.Conv2d(64, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            act_fn,
            nn.UpsamplingNearest2d(size=14),
            nn.Conv2d(16, 8, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(8),
            act_fn,
            nn.UpsamplingNearest2d(size=28),
            nn.Conv2d(8, 1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(1),
            act_fn
        ])

        for layer in encoder:
            layer.apply(initialize_weights)
        for layer in decoder:
            layer.apply(initialize_weights)
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        pl.Trainer()

    def forward(self, x):
        x = self.encoder(x)
        # print(x.size())
        x = self.decoder(x)
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
        out = self.forward(images)
        loss = nn.functional.mse_loss(out, images)
        accuracy = get_accuracy(out, targets)

        return loss, accuracy

    def training_step(self, batch):
        loss, _ = self.general_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.general_step(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", accuracy)
        return loss

    def regenerate_image(self, image):
        image = image[None, None, :, :]
        out = self.forward(image)
        return out

    # implement _end functions

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle =True, batch_size=self.hparams['batch_size'], num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle =True, batch_size=self.hparams['batch_size'], num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle =True, batch_size=self.hparams['batch_size'], num_workers=4)

    def configure_optimizers(self):
        params = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        optim = torch.optim.Adam(params, lr = self.hparams['lr'])
        return optim
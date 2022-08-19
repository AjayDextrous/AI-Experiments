import numpy
import torch
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTWrapper(MNIST):

    def __init__(self,mnist_root,train=True, download=True):
        super().__init__(mnist_root, train=train, download=download)
        self.tensor_transform = transforms.ToTensor()

    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        target_one_hot = torch.zeros(10)
        target_one_hot[target] = 1
        return self.tensor_transform(image), target_one_hot

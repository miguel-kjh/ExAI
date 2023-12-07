import os
from torch import nn
import torch
from torch.nn import functional as F

from models.MLP import MLP

class MNISTModel(MLP):
    def __init__(self, num_classes=10, lr=1e-3):
        """
        MNISTModel is a PyTorch Lightning module for training a model on the MNIST dataset.

        Args:
            num_classes (int): Number of classes in the dataset. Default is 10.
            lr (float): Learning rate for the optimizer. Default is 1e-3.
        """
        super(MNISTModel, self).__init__(num_classes=num_classes, lr=lr)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x, record_activations=False):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
            record_activations (bool): Whether to record intermediate activations. Default is False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x_layer1 = F.relu(x)

        x = self.layer_2(x_layer1)
        x_layer2 = F.relu(x)

        x = self.layer_3(x_layer2)
        x_layer3 = F.log_softmax(x, dim=1)

        if record_activations:
            return x_layer3, [x_layer1, x_layer2, x_layer3]

        return x_layer3
    


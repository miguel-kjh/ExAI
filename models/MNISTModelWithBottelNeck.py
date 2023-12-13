from models.MLP import MLP


from torch import nn
from torch.nn import functional as F


class MNISTModelWithBottelNeck(MLP):
    """
    A custom model for MNIST classification with bottleneck layers.

    Args:
        num_classes (int): The number of classes for classification. Default is 10.
        lr (float): The learning rate for the optimizer. Default is 1e-3.
    """

    def __init__(self, num_classes=10, lr=1e-3):
        super(MNISTModelWithBottelNeck, self).__init__(num_classes=num_classes, lr=lr)
        features = 28*28
        self.layer_1 = nn.Linear(features, 256)
        self.layer_2 = nn.Linear(256, features)
        self.layer_3 = nn.Linear(features, 10)

    def forward(self, x, record_activations=False):
        x = x.view(x.size(0), -1)
        
        x_layer1 = self.layer_1(x)
        x_layer1 = F.relu(x_layer1)
        x_layer2 = self.layer_2(x_layer1)
        x_layer3 = self.layer_3(x_layer2)
        
        x_layer3 = F.log_softmax(x_layer3, dim=1)

        if record_activations:
            return x_layer3, [x_layer1, x_layer2, x_layer3]

        return x_layer3
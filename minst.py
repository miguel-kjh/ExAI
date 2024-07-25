import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
import torchmetrics
import pandas as pd
from tqdm import tqdm

SEED = 2024
pl.seed_everything(SEED)

LEARING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 5
LAMBDA = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {DEVICE}')


def load_data(batch_size=BATCH_SIZE, num_workers=4):
    # Transformaciones para los datos
    transform = transforms.ToTensor()

    # Carga de datos de entrenamiento
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    
    # División entre entrenamiento y validación
    train_size = int(0.8 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = random_split(mnist_train, [train_size, val_size])

    # DataLoader para entrenamiento y validación
    train_loader = DataLoader(mnist_train, batch_size=batch_size, num_workers=num_workers, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, num_workers=num_workers, shuffle=False, persistent_workers=True)

    # Carga de datos de test
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)

    return train_loader, val_loader, test_loader

class MNISTModel(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super(MNISTModel, self).__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)
        self.lr = lr
        self.num_classes = num_classes
        
        # metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')
        self.precision = torchmetrics.Precision(num_classes=self.num_classes, average='macro', task='multiclass')
        self.recall = torchmetrics.Recall(num_classes=self.num_classes, average='macro', task='multiclass')
        self.f1 = torchmetrics.F1Score(num_classes=self.num_classes, average='macro', task='multiclass')
        
    def calculate_loss(self, logits, y):
        return F.nll_loss(logits, y)

    def forward(self, x, record_activations=False):
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
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.calculate_loss(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.calculate_loss(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.calculate_loss(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc(logits, y), prog_bar=True)
        self.log('precision', self.precision(logits, y), prog_bar=True)
        self.log('recall', self.recall(logits, y), prog_bar=True)
        self.log('f1', self.f1(logits, y), prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
class MNISTModelReg(MNISTModel):
    
    def __init__(self, num_classes=10, lr=1e-3, lambda_reg=0.01):
        super(MNISTModelReg, self).__init__(num_classes, lr)
        self.lambda_reg = lambda_reg
    
    def calculate_loss(self, logits, y):
        loss = F.nll_loss(logits, y)
        epsilon = 1e-8
        # Regularización en las capas seleccionadas (ejemplo con fc1 y fc2)
        for layer in [self.layer_1, self.layer_2, self.layer_3]:
            W = layer.weight
            WtW = torch.matmul(W.t(), W)
            det_WtW = torch.det(WtW)
            reg_term = self.lambda_reg * (1.0 / (torch.abs(det_WtW) + epsilon))
            loss += reg_term

        return loss    
    


if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()
    model = MNISTModelReg(lr=LEARING_RATE, lambda_reg=LAMBDA)
    trainer = pl.Trainer(max_epochs=EPOCHS)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
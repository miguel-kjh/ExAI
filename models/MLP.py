from matplotlib import pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F


class MLP(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        """
        Multi-Layer Perceptron model for classification.

        Args:
            num_classes (int): Number of classes for classification. Default is 10.
            lr (float): Learning rate for the optimizer. Default is 1e-3.
        """
        super(MLP, self).__init__()
        self.lr = lr
        self.num_classes = num_classes

        # metrics
        self.train_acc = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=self.num_classes, task='multiclass')
        self.precision = torchmetrics.Precision(num_classes=self.num_classes, average='macro', task='multiclass')
        self.recall = torchmetrics.Recall(num_classes=self.num_classes, average='macro', task='multiclass')
        self.f1 = torchmetrics.F1Score(num_classes=self.num_classes, average='macro', task='multiclass')

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch (tuple): Tuple containing input tensor and target tensor.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc(logits, y), prog_bar=True)
        return loss
    
    def custom_histogram_adder(self):
        try:
            for name,params in self.named_parameters():           
                self.logger.experiment.add_histogram(name,params,self.current_epoch)
        except Exception as e:
            print("Failed to save weights histogram: {}".format(e))

    def makegrid(output,numrows):
        outer=(torch.Tensor.cpu(output).detach())
        plt.figure(figsize=(20,5))
        b=np.array([]).reshape(0,outer.shape[2])
        c=np.array([]).reshape(numrows*outer.shape[2],0)
        i=0
        j=0
        while(i < outer.shape[1]):
            img=outer[0][i]
            b=np.concatenate((img,b),axis=0)
            j+=1
            if(j==numrows):
                c=np.concatenate((c,b),axis=1)
                b=np.array([]).reshape(0,outer.shape[2])
                j=0
            i+=1
        return c
    
    def on_train_start(self):
        try:
            sample_input = torch.rand((1,1,28,28)).to(self.device)
            self.logger.experiment.add_graph(self, sample_input)
        except Exception as e:
            print("Failed to save model graph: {}".format(e))
    
    def showActivations(self, x):
        pass

    def on_train_epoch_end(self) -> None:
        self.custom_histogram_adder()

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch (tuple): Tuple containing input tensor and target tensor.
            batch_idx (int): Index of the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step of the model.

        Args:
            batch (tuple): Tuple containing input tensor and target tensor.
            batch_idx (int): Index of the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc(logits, y), prog_bar=True)
        self.log('precision', self.precision(logits, y), prog_bar=True)
        self.log('recall', self.recall(logits, y), prog_bar=True)
        self.log('f1', self.f1(logits, y), prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
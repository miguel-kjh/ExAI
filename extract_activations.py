import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from MNISTModel import MNISTModel

MODEL_PATH = 'model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    # load only one image of the test set
    transform = transforms.ToTensor()
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)
    return test_loader

def main():
    model = MNISTModel.load_from_checkpoint('lightning_logs/version_6/checkpoints/epoch=4-step=3750.ckpt')
    model.eval()
    test_loader = load_data()
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        _, activations = model(x, record_activations=True)
        print(activations[0])
        break

if __name__ == "__main__":
    main()
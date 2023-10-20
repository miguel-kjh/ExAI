import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Definir una red neuronal simple
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Hiperparámetros
num_epochs = 10
batch_size = 64
learning_rate = 0.001
few_shot_examples_per_class = 3 # K
number_of_class = 10

# Transformaciones de datos para MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Cargar el conjunto de datos MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Definir DataLoader para entrenamiento
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Entrenar modelos para cada clase con pocos ejemplos
models = []
for class_label in range(number_of_class):
    # Filtrar ejemplos de la clase actual
    class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_label]
    subset_indices = class_indices[:few_shot_examples_per_class]

    # Crear un DataLoader para pocos ejemplos de la clase
    few_shot_data = Subset(train_dataset, subset_indices)
    few_shot_loader = DataLoader(few_shot_data, batch_size=batch_size, shuffle=True)

    # Crear y entrenar un modelo para la clase actual
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Episodio de entrenamiento para la clase actual
    for epoch in tqdm(range(num_epochs), desc='Training model for class {}'.format(class_label)):
        for images, labels in few_shot_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    models.append(model)

# Evaluar el rendimiento en el conjunto de prueba
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataset:
        outputs_list = [model(images) for model in models]
        outputs = torch.stack(outputs_list, dim=0).mean(dim=0)
        _, predicted = torch.max(outputs.data, 0)
        total += 1
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print('Accuracy on test set: {:.2%}'.format(accuracy))

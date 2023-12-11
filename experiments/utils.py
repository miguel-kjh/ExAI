import os
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
# get confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



class FilteredMNIST(Dataset):
    def __init__(self, mnist_dataset, classes_to_include):
        self.mnist_dataset = mnist_dataset
        self.filtered_indices = [i for i, (image, label) in enumerate(mnist_dataset) if label in classes_to_include]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        return self.mnist_dataset[self.filtered_indices[idx]]

def load_data_filtered(batch_size, classes_to_include, num_workers=4) -> tuple:
    # Transformaciones para los datos
    transform = transforms.ToTensor()

    # Carga de datos de entrenamiento
    mnist_train = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transform)

    # filtrar el dataset para incluir solo clases específicas
    filtered_train_dataset = FilteredMNIST(mnist_train, classes_to_include)

    # División entre entrenamiento y validación
    train_size = int(0.8 * len(filtered_train_dataset))
    val_size = len(filtered_train_dataset) - train_size
    mnist_train, mnist_val = random_split(filtered_train_dataset, [train_size, val_size])

    # DataLoader para entrenamiento y validación
    train_loader = DataLoader(mnist_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Carga de datos de test
    mnist_test = datasets.MNIST(os.getcwd(), train=False, download=True, transform=transform)
    
    # Filtrar el dataset de test para incluir solo clases específicas
    filtered_test_dataset = FilteredMNIST(mnist_test, classes_to_include)

    # DataLoader para el conjunto de datos filtrado
    test_loader = DataLoader(filtered_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


def find_low_activation_neurons(df, number1, number2, threshold=0.5):
    # 1. Filtrar los datos por número
    df_num1 = df[df['Number'] == number1].drop(columns=['Number'])
    df_num2 = df[df['Number'] == number2].drop(columns=['Number'])

    # 2. Calcular la media de las activaciones
    mean_num1 = df_num1.mean()
    mean_num2 = df_num2.mean()

    # 3. Identificar las neuronas con media inferior al 50% del valor máximo de la media
    threshold_num1 = threshold * mean_num1.max()
    threshold_num2 = threshold * mean_num2.max()
    
    low_neurons_num1 = mean_num1[mean_num1 < threshold_num1].index
    low_neurons_num2 = mean_num2[mean_num2 < threshold_num2].index

    # 4. Calcular la intersección de las neuronas
    intersection_neurons = set(low_neurons_num1).intersection(set(low_neurons_num2))

    # get index form str
    intersection_neurons = [int(i.split('Neuron')[1])-1 for i in intersection_neurons]

    # 5. Retornar la lista de neuronas en la intersección
    return list(intersection_neurons)



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.


    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    fontsize=20,
                    color="white" if cm[i, j] > thresh else "black")


    fig.tight_layout()
    return ax


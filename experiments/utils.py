import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
# get confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA


class FilteredMNIST(Dataset):
    """
    A custom dataset class that filters the MNIST dataset based on specified classes.

    Args:
        mnist_dataset (Dataset): The original MNIST dataset.
        classes_to_include (list): A list of classes to include in the filtered dataset.

    Attributes:
        mnist_dataset (Dataset): The original MNIST dataset.
        filtered_indices (list): A list of indices corresponding to the filtered dataset.

    Methods:
        __len__(): Returns the length of the filtered dataset.
        __getitem__(idx): Returns a specific item from the filtered dataset.

    """

    def __init__(self, mnist_dataset, classes_to_include):
        self.mnist_dataset = mnist_dataset
        self.filtered_indices = [i for i, (image, label) in enumerate(mnist_dataset) if label in classes_to_include]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        return self.mnist_dataset[self.filtered_indices[idx]]

def load_data_filtered(batch_size, classes_to_include, num_workers=4) -> tuple:
    """
    Load filtered MNIST data and create data loaders for training, validation, and testing.

    Args:
        batch_size (int): The batch size for the data loaders.
        classes_to_include (list): The list of classes to include in the filtered dataset.
        num_workers (int, optional): The number of worker threads for data loading. Defaults to 4.

    Returns:
        tuple: A tuple containing the data loaders for training, validation, and testing.
    """
    transform = transforms.ToTensor()

    mnist_train = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transform)

    filtered_train_dataset = FilteredMNIST(mnist_train, classes_to_include)

    train_size = int(0.8 * len(filtered_train_dataset))
    val_size = len(filtered_train_dataset) - train_size
    mnist_train, mnist_val = random_split(filtered_train_dataset, [train_size, val_size])

    train_loader = DataLoader(mnist_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    mnist_test = datasets.MNIST(os.getcwd(), train=False, download=True, transform=transform)
    
    filtered_test_dataset = FilteredMNIST(mnist_test, classes_to_include)

    test_loader = DataLoader(filtered_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


def find_low_activation_neurons(df, number1, number2, threshold=0.5):
    """
    Finds the low activation neurons that are below the given threshold for both number1 and number2.

    Parameters:
    - df (DataFrame): The input DataFrame containing activation values.
    - number1 (int): The first number to compare.
    - number2 (int): The second number to compare.
    - threshold (float): The threshold value for determining low activation.

    Returns:
    - intersection_neurons (list): A list of indices of the low activation neurons that are below the threshold for both number1 and number2.
    """
    df_num1 = df[df['Number'] == number1].drop(columns=['Number'])
    df_num2 = df[df['Number'] == number2].drop(columns=['Number'])

    mean_num1 = df_num1.mean()
    mean_num2 = df_num2.mean()

    threshold_num1 = threshold * mean_num1.max()
    threshold_num2 = threshold * mean_num2.max()
    
    low_neurons_num1 = mean_num1[mean_num1 < threshold_num1].index
    low_neurons_num2 = mean_num2[mean_num2 < threshold_num2].index

    intersection_neurons = set(low_neurons_num1).intersection(set(low_neurons_num2))

    intersection_neurons = [int(i.split('Neuron')[1])-1 for i in intersection_neurons]

    return intersection_neurons



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


def remove_neurons_from_layer(layer, neurons_to_remove):
    """ Elimina neuronas de una capa específica. """
    new_weights = torch.cat([layer.weight.data[i:i+1] for i in range(layer.weight.data.size(0)) if i not in neurons_to_remove])
    new_bias = torch.cat([layer.bias.data[i:i+1] for i in range(layer.bias.data.size(0)) if i not in neurons_to_remove])
    return nn.Linear(new_weights.size(1), new_weights.size(0)), new_weights, new_bias

def adjust_next_layer(layer, neurons_to_remove_from_previous_layer):
    """ Ajusta la siguiente capa eliminando las columnas de los pesos. """
    new_weights = torch.cat([layer.weight.data[:, i:i+1] for i in range(layer.weight.data.size(1)) if i not in neurons_to_remove_from_previous_layer], 1)
    return nn.Linear(new_weights.size(1), new_weights.size(0)), new_weights

def weight_statistics(model, layer_name):
    weights = getattr(model, layer_name).weight.data
    stats = {
        "mean": torch.mean(weights).item(),
        "std_dev": torch.std(weights).item(),
        "max": torch.max(weights).item(),
        "min": torch.min(weights).item()
    }
    return stats

def plot_weights(model, layer_name):
    weights = getattr(model, layer_name).weight.data
    plt.matshow(weights)
    plt.colorbar()
    plt.title(f'Weights {layer_name}')
    plt.xlabel('Input')
    plt.ylabel('Nurons')
    plt.show()

def plot_weight_histograms(weights1, weights2, layer_index, label1='Model 1', label2='Model 2'):
    plt.figure(figsize=(12, 6))

    # Layer 1
    plt.subplot(1, 2, 1)
    plt.hist(weights1[layer_index].flatten(), bins=50, alpha=0.5, label=label1)
    plt.hist(weights2[layer_index].flatten(), bins=50, alpha=0.5, label=label2)
    plt.title(f'Histogram of Weights - Layer {layer_index + 1}')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Flatten the weights for PCA
# Each neuron's weights become a single vector

def calculate_pca(weights, n_components=2):
    flattened_weights = weights.reshape(weights.shape[0], -1)

    # Apply PCA to reduce the dimensions to 2 for visualization
    pca = PCA(n_components=n_components)
    reduced_weights = pca.fit_transform(flattened_weights)

    return reduced_weights[:flattened_weights.shape[0], :]

def plot_pca(weights1, weights2, layer_index):

    # Calculate PCA for each model
    reduced_weights_model_1 = calculate_pca(weights1[layer_index])
    reduced_weights_model_2 = calculate_pca(weights2[layer_index])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_weights_model_1[:, 0], reduced_weights_model_1[:, 1], label='Model Original', alpha=0.7, marker='o', c='b')
    plt.scatter(reduced_weights_model_2[:, 0], reduced_weights_model_2[:, 1], label='Model Pruning', alpha=0.7, marker='o' , c='r')
    plt.title(f'PCA of Weights in the {layer_index+1} Layer')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
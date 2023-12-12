import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
# copy
import copy


# Definición de la clase MNISTModel
class MNISTModel(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super(MNISTModel, self).__init__()
        self.layer_1 = nn.Linear(4, 3)
        self.layer_2 = nn.Linear(3, 3)
        self.layer_3 = nn.Linear(3, num_classes)
        self.lr = lr

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

# Función para eliminar neuronas de una capa
def remove_neurons(layer, neurons_to_remove):
    new_weights = layer.weight.data.clone()
    new_bias = layer.bias.data.clone()

    # Eliminar los pesos y sesgos de las neuronas especificadas
    new_weights = torch.cat([new_weights[i:i+1] for i in range(new_weights.size(0)) if i not in neurons_to_remove])
    print(new_weights.size(0))
    new_bias = torch.cat([new_bias[i:i+1] for i in range(new_bias.size(0)) if i not in neurons_to_remove])

    # Crear una nueva capa con los pesos y sesgos actualizados
    new_layer = nn.Linear(layer.in_features, new_weights.size(0))
    new_layer.weight.data = new_weights
    new_layer.bias.data = new_bias

    return new_layer

def plot_weights(original_weights, modified_weights):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Gráfico de los pesos originales
    cax1 = axes[0].matshow(original_weights, aspect='auto')
    fig.colorbar(cax1, ax=axes[0])
    axes[0].set_title('Original Weights')
    axes[0].set_xlabel('Input Neurons')
    axes[0].set_ylabel('Output Neurons')

    # Anotar valores en el gráfico
    for (i, j), val in np.ndenumerate(original_weights):
        axes[0].text(j, i, f'{val:.2f}', ha='center', va='center')

    # Gráfico de los pesos modificados
    cax2 = axes[1].matshow(modified_weights, aspect='auto')
    fig.colorbar(cax2, ax=axes[1])
    axes[1].set_title('Modified Weights')
    axes[1].set_xlabel('Input Neurons')
    axes[1].set_ylabel('Output Neurons')

    # Anotar valores en el gráfico
    for (i, j), val in np.ndenumerate(modified_weights):
        axes[1].text(j, i, f'{val:.2f}', ha='center', va='center')

    plt.show()

# Creación del modelo original
original_model = MNISTModel()
modified_model = copy.deepcopy(original_model)

# Imprimir el modelo original
print("\nModelo original:")
print(original_model)

# Neuronas a eliminar
# Supongamos que original_model es tu modelo y neurons_to_remove es una lista de listas de neuronas a eliminar en cada capa
layer_1_neurons_to_remove = [0, 2]  # Ejemplo
layer_2_neurons_to_remove = [0]  # Ejemplo

def remove_neurons_from_layer(layer, neurons_to_remove):
    """ Elimina neuronas de una capa específica. """
    new_weights = torch.cat([layer.weight.data[i:i+1] for i in range(layer.weight.data.size(0)) if i not in neurons_to_remove])
    new_bias = torch.cat([layer.bias.data[i:i+1] for i in range(layer.bias.data.size(0)) if i not in neurons_to_remove])
    return nn.Linear(new_weights.size(1), new_weights.size(0)), new_weights, new_bias

def adjust_next_layer(layer, neurons_to_remove_from_previous_layer):
    """ Ajusta la siguiente capa eliminando las columnas de los pesos. """
    new_weights = torch.cat([layer.weight.data[:, i:i+1] for i in range(layer.weight.data.size(1)) if i not in neurons_to_remove_from_previous_layer], 1)
    return nn.Linear(new_weights.size(1), new_weights.size(0)), new_weights

new_layer_1, new_weights_1, new_bias_1 = remove_neurons_from_layer(modified_model.layer_1, layer_1_neurons_to_remove)
new_layer_2, new_weights_2 = adjust_next_layer(modified_model.layer_2, layer_1_neurons_to_remove)

# Aplica los nuevos pesos y sesgos
modified_model.layer_1 = new_layer_1
modified_model.layer_1.weight.data = new_weights_1
modified_model.layer_1.bias.data = new_bias_1

modified_model.layer_2 = new_layer_2
modified_model.layer_2.weight.data = new_weights_2
# Los sesgos de layer_2 no cambian a menos que también elimines neuronas de layer_2

new_layer_2, new_weights_2, new_bias_2 = remove_neurons_from_layer(modified_model.layer_2, layer_2_neurons_to_remove)

# Aplicar los nuevos pesos y sesgos a layer_2
modified_model.layer_2 = new_layer_2
modified_model.layer_2.weight.data = new_weights_2
modified_model.layer_2.bias.data = new_bias_2

# Ajustar layer_3 en consecuencia
# Supongamos que layer_3 no tiene neuronas eliminadas directamente, pero sus pesos deben ajustarse debido a los cambios en layer_2
new_layer_3, new_weights_3 = adjust_next_layer(modified_model.layer_3, layer_2_neurons_to_remove)

# Aplicar los nuevos pesos a layer_3
modified_model.layer_3 = new_layer_3
modified_model.layer_3.weight.data = new_weights_3

print("\nModelo con neuronas eliminadas:")
print(modified_model)

plot_weights(original_model.layer_1.weight.data, modified_model.layer_1.weight.data)
plot_weights(original_model.layer_2.weight.data, modified_model.layer_2.weight.data)


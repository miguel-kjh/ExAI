from typing import List
import pandas as pd
import torch
import torch.nn as nn


import copy

from models.MNISTModel import MNISTModel
from models.MNISTModelWithBottelNeck import MNISTModelWithBottelNeck


class PruningMLP:

    def __init__(self, numbers: List[str]) -> None:
        self.numbers = numbers

    def _get_layer_columns(self, layer: str, columns: List[str]):
        layer_columns = ['Number']
        layer_columns += [c for c in columns if layer in c]
        return layer_columns
    
    def _find_low_activation_neurons(self, df: pd.DataFrame,  threshold: float = 0.5):

        low_neurons_sets = []

        for number in self.numbers:
            df_num = df[df['Number'] == number].drop(columns=['Number'])
            mean_num = df_num.mean()
            threshold_num = threshold * mean_num.max()
            low_neurons = mean_num[mean_num < threshold_num].index
            low_neurons_sets.append(set(low_neurons))

        intersection_neurons = set.intersection(*low_neurons_sets)
        intersection_neurons = [int(i.split('Neuron')[1])-1 for i in intersection_neurons]

        return intersection_neurons
    
    def remove_neurons_from_layer(self, layer, neurons_to_remove):
        new_weights = torch.cat([layer.weight.data[i:i+1] for i in range(layer.weight.data.size(0)) if i not in neurons_to_remove])
        new_bias = torch.cat([layer.bias.data[i:i+1] for i in range(layer.bias.data.size(0)) if i not in neurons_to_remove])
        return nn.Linear(new_weights.size(1), new_weights.size(0)), new_weights, new_bias

    def adjust_next_layer(self, layer, neurons_to_remove_from_previous_layer):
        new_weights = torch.cat([layer.weight.data[:, i:i+1] for i in range(layer.weight.data.size(1)) if i not in neurons_to_remove_from_previous_layer], 1)
        return nn.Linear(new_weights.size(1), new_weights.size(0)), new_weights

    def prune(self, model: nn.Module, df_activation: pd.DataFrame, thresholds: dict) -> nn.Module:
        mask_indices = {}
        model_copy = copy.deepcopy(model)

        for layer_name, threshold in thresholds.items():
            activation_for_layers = df_activation[self._get_layer_columns(layer_name, df_activation.columns)]
            mask_indices[layer_name] = self._find_low_activation_neurons(activation_for_layers, threshold)


        # Aplicar la poda y ajustar las capas
        for layer_name, mask in mask_indices.items():
            if len(mask) > 0:
                # Asumimos que las capas están nombradas de forma estándar (layer_1, layer_2, etc.)
                current_layer = getattr(model, layer_name)
                next_layer_name = f"layer_{int(layer_name.split('_')[1]) + 1}"

                # Eliminar neuronas de la capa actual
                new_layer, new_weights, new_biases = self.remove_neurons_from_layer(current_layer, mask)
                setattr(model, layer_name, new_layer)
                model._modules[layer_name].weight.data = new_weights
                model._modules[layer_name].bias.data = new_biases

                # Ajustar la siguiente capa si existe
                if hasattr(model, next_layer_name):
                    next_layer = getattr(model, next_layer_name)
                    new_next_layer, new_next_weights = self.adjust_next_layer(next_layer, mask)
                    setattr(model, next_layer_name, new_next_layer)
                    model._modules[next_layer_name].weight.data = new_next_weights

        return model
    
# main
if __name__ == "__main__":
    checkpoint_ffn = 'experiments/checkpoints/mnist_model.ckpt'
    model = MNISTModel.load_from_checkpoint(checkpoint_ffn)
    model.eval()
    print(model)

    p = PruningMLP([0,1])
    df = pd.read_csv("experiments/activations/activations_minist_model.csv")
    model_2 = p.prune(model, df, {'layer_1': 0.5, 'layer_2': 0.25})
    print(model_2)

    """bottelnet_checkpoint = 'experiments/checkpoints/mnist_model_bottel_neck.ckpt'
    model_bottelnet = MNISTModelWithBottelNeck.load_from_checkpoint(bottelnet_checkpoint)
    model_bottelnet.eval()
    print(model_bottelnet)

    df = pd.read_csv("experiments/activations/activations_minist_model_bottel_neck.csv")
    model_bottelnet_2 = p.prune(model_bottelnet, df, {'layer_2': 0.25, 'layer_3': 0.25})
    print(model_bottelnet_2)"""

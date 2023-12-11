import torch
from models.MNISTModel import MNISTModel
from torch.nn import functional as F


class PlasticityMNISTModel(MNISTModel):

    def __init__(
            self, 
            num_classes=10, 
            lr=1e-3, 
            mask_indices_layer1=None, 
            mask_indices_layer2=None, 
            mask_indices_layer3=None,
        ):

        super(PlasticityMNISTModel, self).__init__(num_classes=num_classes, lr=lr)

        self.mask_indices_layer1 = mask_indices_layer1
        self.mask_indices_layer2 = mask_indices_layer2
        self.mask_indices_layer3 = mask_indices_layer3

    def apply_neuron_mask(self, layer_output, indices_to_zero):
        mask = torch.ones(layer_output.size(1), device=layer_output.device)
        mask[indices_to_zero] = 0
        return layer_output * mask
    
    def forward(self, x, record_activations=False):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x_layer1 = F.relu(x)

        # Aplicar máscara a la capa 1
        if self.mask_indices_layer1 is not None:
            x_layer1 = self.apply_neuron_mask(x_layer1, self.mask_indices_layer1)

        x = self.layer_2(x_layer1)
        x_layer2 = F.relu(x)

        # Aplicar máscara a la capa 2
        if self.mask_indices_layer2 is not None:
            x_layer2 = self.apply_neuron_mask(x_layer2, self.mask_indices_layer2)

        x = self.layer_3(x_layer2)
        x_layer3 = F.log_softmax(x, dim=1)

        # Aplicar máscara a la capa 3
        if self.mask_indices_layer3 is not None:
            x_layer3 = self.apply_neuron_mask(x_layer3, self.mask_indices_layer3)

        if record_activations:
            return x_layer3, [x_layer1, x_layer2, x_layer3]

        return x_layer3 
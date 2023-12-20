from GPT2LayerWithFFNOutput import GPT2LayerWithFFNOutput


import torch.nn as nn
from transformers import GPT2Model


class GPT2ModelWithFFNOutput(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPT2LayerWithFFNOutput(config) for _ in range(config.n_layer)])
import pickle
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

# no warnings
import warnings
warnings.filterwarnings("ignore")

class GPT2MLPActivations(GPT2MLP):

    def __init__(self,  intermediate_size, config):
        super().__init__(intermediate_size, config)
        self._activations = []

    @property
    def activations(self):
        return self._activations

    def _normalize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x_mean = torch.mean(x, dim=1, keepdim=True)[0]
        return x_mean / torch.norm(x_mean, p=2, dim=-1, keepdim=True)
    
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        self._activations.append(self._normalize(hidden_states).cpu().detach().numpy())
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2LayerWithFFNOutput(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.mlp = GPT2MLPActivations(inner_dim, config)

class GPT2ModelWithFFNOutput(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPT2LayerWithFFNOutput(config) for _ in range(config.n_layer)])

class LLMHeadModelWithFFNOutput(GPT2LMHeadModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelWithFFNOutput(config)

    def get_activation_ffn(self):
        return [
            self.transformer.h[i].mlp.activations
            for i in range(len(self.transformer.h))
        ]


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
num_layers = config.n_layer
model = LLMHeadModelWithFFNOutput(config)

# Asegúrate de que el modelo esté en modo de evaluación
model.eval()

# Asegúrate de que las operaciones se realicen en el dispositivo correcto
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = ["128 ", "35 ", "47 ", "199 "]
recollection = {}

for t in text:
    input_ids = torch.tensor([tokenizer.encode(t)]).to(device)
    tokens = tokenizer.tokenize(t)
    print(tokens)

    outputs_tokens = model.generate(input_ids, max_length=2, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs_tokens[0], skip_special_tokens=True)
    print(output_text)

    activation_ffn = model.get_activation_ffn()
    print(len(activation_ffn))
    print(activation_ffn[0][0].shape)
    
    recollection[t] = {
        "output_text": output_text
    }

    for i in range(num_layers):
        recollection[t][i] = activation_ffn[i][0]

print(recollection)

#save in pickle
with open('recollection.pickle', 'wb') as handle:
    pickle.dump(recollection, handle, protocol=pickle.HIGHEST_PROTOCOL)


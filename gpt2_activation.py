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
        self.activations = []

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        self.activations.append(hidden_states.detach().cpu().numpy())
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
        return {
            f"Layer {i}": self.transformer.h[i].mlp.activations
            for i in range(len(self.transformer.h))
        }


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
model = LLMHeadModelWithFFNOutput(config)

# Asegúrate de que el modelo esté en modo de evaluación
model.eval()

# Asegúrate de que las operaciones se realicen en el dispositivo correcto
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = ["Nice to meet you", "Hello!", "How are you?"]
recollection = {}

for t in tqdm(text, desc="Text"):
    input_ids = torch.tensor([tokenizer.encode(t)]).to(device)

    outputs_tokens = model.generate(input_ids, max_length=1)
    output_text = tokenizer.decode(outputs_tokens[0], skip_special_tokens=True)
    #print(output_text)

    max_activation_ffn = model.get_activation_ffn()
    #print(max_activation_ffn)
    
    recollection[output_text] = max_activation_ffn

for key, value in recollection.items():
    print(key)
    print(value['Layer 0'][0].shape)

#save max_activation_ffn in pkl file
"""with open('max_activation_ffn.pkl', 'wb') as f:
    pickle.dump(recollection, f)"""


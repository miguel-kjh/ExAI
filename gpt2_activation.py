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

class GPT2MLPWithFFNOutput(GPT2MLP):

    def __init__(self,  intermediate_size, config):
        super().__init__(intermediate_size, config)
        self.max_activation_ffn = {
            "c_fc": {
                "max_index": [],
                "max_weight": [],
            },
            "c_proj": {
                "max_index": [],
                "max_weight": [],
            },
        }

    def _normalize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x / ((x ** 2).sum(dim=-1, keepdim=True) ** 0.5)
    
    def _get_max_activation(self, x: torch.FloatTensor) -> Tuple[int, int]:
        norm_x = self._normalize(x)
        max_val = torch.max(norm_x)
        max_pos = torch.where(norm_x == max_val)
        return (max_pos[0].item(), max_pos[1].item(), max_pos[2].item())
    
    def _get_max_info(self, weigths: torch.FloatTensor, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        max_index = self._get_max_activation(x)
        max_weight = weigths[:, max_index[2]]
        return max_index, max_weight
    
    def _recollect_max_info(self, weigths: torch.FloatTensor, x: torch.FloatTensor, layer_name: str) -> None:
        max_index, max_weight = self._get_max_info(weigths, x)
        self.max_activation_ffn[layer_name]['max_index'] = max_index
        self.max_activation_ffn[layer_name]['max_weight'] = max_weight.cpu().numpy()

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        self._recollect_max_info(self.c_fc.weight, hidden_states, 'c_fc')
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        self._recollect_max_info(self.c_proj.weight, hidden_states, 'c_proj')
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2LayerWithFFNOutput(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.mlp = GPT2MLPWithFFNOutput(inner_dim, config)

class GPT2ModelWithFFNOutput(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPT2LayerWithFFNOutput(config) for _ in range(config.n_layer)])

class LLMHeadModelWithFFNOutput(GPT2LMHeadModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelWithFFNOutput(config)

    def get_max_activation_ffn(self):
        return {
            f"Layer {i}": self.transformer.h[i].mlp.max_activation_ffn
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
    print(output_text)

    max_activation_ffn = model.get_max_activation_ffn()
    
    recollection[output_text] = max_activation_ffn

print(recollection.keys())

#save max_activation_ffn in pkl file
with open('max_activation_ffn.pkl', 'wb') as f:
    pickle.dump(recollection, f)


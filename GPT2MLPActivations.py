import torch
import numpy as np
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP


from typing import Optional, Tuple


class GPT2MLPActivations(GPT2MLP):

    def __init__(self,  intermediate_size, config):
        super().__init__(intermediate_size, config)
        self._activations: np.ndarray = None

    @property
    def activations(self) -> np.ndarray:
        return self._activations
        
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        self._activations = hidden_states.cpu().detach().numpy()
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
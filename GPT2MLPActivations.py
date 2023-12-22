import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP


from typing import Optional, Tuple


class GPT2MLPActivations(GPT2MLP):

    def __init__(self,  intermediate_size, config):
        super().__init__(intermediate_size, config)
        self._activations = []

    @property
    def activations(self):
        return self._activations

    def _normalize(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x_mean = torch.mean(x, dim=1, keepdim=True)[0]
        # normalize between 0 and 1
        x_min = torch.min(x_mean, dim=1, keepdim=True)[0]
        x_max = torch.max(x_mean, dim=1, keepdim=True)[0]
        x_norm = (x_mean - x_min) / (x_max - x_min)
        return x_norm
        
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        self._activations.append(self._normalize(hidden_states).cpu().detach().numpy())
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
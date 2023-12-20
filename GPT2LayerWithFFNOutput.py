from GPT2MLPActivations import GPT2MLPActivations


from transformers.models.gpt2.modeling_gpt2 import GPT2Block


class GPT2LayerWithFFNOutput(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.mlp = GPT2MLPActivations(inner_dim, config)
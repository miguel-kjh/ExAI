from GPT2ModelWithFFNOutput import GPT2ModelWithFFNOutput


from transformers import GPT2LMHeadModel


class LLMHeadModelWithFFNOutput(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelWithFFNOutput(config)

    def get_activation_ffn(self):
        return [
            self.transformer.h[i].mlp.activations
            for i in range(len(self.transformer.h))
        ]
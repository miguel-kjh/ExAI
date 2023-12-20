import pickle
import torch
from transformers import GPT2Tokenizer, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

# no warnings
import warnings
from LLMHeadModelWithFFNOutput import LLMHeadModelWithFFNOutput

warnings.filterwarnings("ignore")

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


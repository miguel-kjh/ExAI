import pickle
import pandas as pd
import torch
from tqdm import tqdm
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
num_neurons = config.hidden_size*4
model = LLMHeadModelWithFFNOutput(config)


# Asegúrate de que el modelo esté en modo de evaluación
model.eval()

# Asegúrate de que las operaciones se realicen en el dispositivo correcto
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = ["Obama ", "Donal ", "Trump ", "Joe "]

columns = ["Input_text", "Output_text"]
for layer_index in range(num_layers):
    columns += [f'layer_{layer_index+1}_Neuron{i+1}' for i in range(num_neurons)]

df = pd.DataFrame(index=range(len(text)), columns=columns)
                  
for t in tqdm(text, desc="Texts"): 
    input_ids = torch.tensor([tokenizer.encode(t)]).to(device)
    #tokens = tokenizer.tokenize(t)

    outputs_tokens = model.generate(input_ids, max_length=2, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs_tokens[0], skip_special_tokens=True)

    activation_ffn = model.get_activation_ffn()
    
    row = {
        "Input_text": t,
        "Output_text": output_text
    }

    for layer_index in range(num_layers):
        for neuron_index in range(num_neurons):
            row[f'layer_{layer_index+1}_Neuron{neuron_index+1}'] = activation_ffn[layer_index][0][0][neuron_index]

    df.loc[len(df)] = row


#save in pickle
with open('recollection.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

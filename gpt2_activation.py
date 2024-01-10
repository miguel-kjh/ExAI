import pickle
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Config
from LLMHeadModelWithFFNOutput import LLMHeadModelWithFFNOutput

# Configuración del modelo y el tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
num_layers = config.n_layer
num_neurons = config.hidden_size * 4
max_length = 15
folder = os.path.join('experiments', "activations")
file = os.path.join(folder, "gpt2.pickle")
model = LLMHeadModelWithFFNOutput(config)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Datos de entrada
text = [
    "The war lasted from the year 1732 to the year 17"
]  # Ejemplos adicionales

# Procesamiento de textos
activations = []
for t in tqdm(text, desc="Texts"):
    input_ids = torch.tensor([tokenizer.encode(t)]).to(device)
    outputs_tokens = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs_tokens[0], skip_special_tokens=True)
    activation_ffn = model.get_activation_ffn()

    # Almacenar la activación en una lista
    activations.append(activation_ffn)

# Convertir la lista de arrays de NumPy en un solo array
activation_ffn_array = np.stack(activations)

# Redimensionar el array para obtener la forma deseada
activation_ffn_array = activation_ffn_array.reshape(num_layers, len(text), -1, num_neurons)
print(activation_ffn_array.shape)

# save in pickle file
os.makedirs(folder, exist_ok=True)
with open(file, 'wb') as f:
    pickle.dump(activation_ffn_array, f)
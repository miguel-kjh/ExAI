import pickle
import os
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
max_length = 2
folder = os.path.join('experiments', "activations")
file = os.path.join(folder, "gpt2.pickle")
model = LLMHeadModelWithFFNOutput(config)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Datos de entrada
text = ['The dog is', 'The cat is', 'The dog is cute', 'The cat is cute', 'The dog is cute and', 'The cat is cute and', 
        'The dog is cute and the', 'The cat is cute and the', 'The dog is cute and the cat', 'The cat is cute and the dog',
          'The dog is cute and the cat is', 'The cat is cute and the dog is', 'The dog is cute and the cat is cute', 
          'The cat is cute and the dog is cute', 'The dog is cute and the cat is cute and', 'The cat is cute and the dog is cute and', 
          'The dog is cute and the cat is cute and the', 'The cat is cute and the dog is cute and the', 'The dog is cute and the cat is cute and the dog', 
          'The cat is cute and the dog is cute and the cat', 'The dog is cute and the cat is cute and the dog is', 'The cat is cute and the dog is cute and the cat is', 
          'The dog is cute and the cat is cute and the dog is cute', 'The cat is cute and the dog is cute and the cat is cute', 
          'The dog is cute and the cat is cute and the dog is cute and', 'The cat is cute and the dog is cute and the cat is cute and', 
          'The dog is cute and the cat is cute and the dog is cute and the', 'The cat is cute and the dog is cute and the cat is cute and the', 
          'The dog is cute and the cat is cute and the dog is cute and the cat', 'The cat is cute and the dog is cute and the cat is cute and the dog', 
          'The dog is cute and the cat is cute and the dog is cute and the cat is', 'The cat is cute and the dog is cute and the cat is cute and the dog is', 
          'The dog is cute and the cat is cute and the dog is cute and the cat is cute', 'The cat is cute and the dog is cute and the cat is cute and the dog is cute', 
          'The dog is cute and the cat is cute and the dog is cute and the cat is cute and', 'The cat is cute and the dog is cute and the cat is cute and the dog is cute and', 
          'The dog is cute and the cat is cute and the dog is cute and the cat is cute and']

# Preparar columnas para el DataFrame
columns = ["Input_text", "Output_text"] + [f'layer_{i+1}_Neuron{j+1}' for i in range(num_layers) for j in range(num_neurons)]
rows = []

# Procesamiento de textos
for t in tqdm(text, desc="Texts"):
    input_ids = torch.tensor([tokenizer.encode(t)]).to(device)
    outputs_tokens = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs_tokens[0], skip_special_tokens=True)
    activation_ffn = model.get_activation_ffn()

    row = {"Input_text": t, "Output_text": output_text}
    for layer_index in range(num_layers):
        for neuron_index in range(num_neurons):
            row[f'layer_{layer_index+1}_Neuron{neuron_index+1}'] = activation_ffn[layer_index][0][0][neuron_index]
    rows.append(row)

# Crear DataFrame y guardar en formato pickle
df = pd.DataFrame(rows, columns=columns)
with open(file, 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


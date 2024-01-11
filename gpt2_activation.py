import pickle
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Config
from LLMHeadModelWithFFNOutput import LLMHeadModelWithFFNOutput

test_file = os.path.join('datasets', 'test_dataset.txt')

# Configuración del modelo y el tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
num_layers = config.n_layer
num_neurons = config.hidden_size * 4
max_length = 19
folder = os.path.join('experiments', "activations")
file = os.path.join(folder, "gpt2.pickle")
model = LLMHeadModelWithFFNOutput.from_pretrained('gpt2-finetuned', config=config)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Datos de entrada, lee los datos de test es un txt
with open(test_file, encoding="utf-8") as file:
    text = file.readlines()

# separar por :
index_number = -3
prompts = [t[:index_number].strip() for t in text]
numbers = [t[index_number:].strip() for t in text]

# Procesamiento de textos
activations = []
accuracy = 0
for prompt, number in tqdm(zip(prompts, numbers), desc="Testing..."):
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    outputs_tokens = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs_tokens[0], skip_special_tokens=True)
    #activation_ffn = model.get_activation_ffn()

    # Almacenar la activación en una lista
    #activations.append(activation_ffn)
    output_text = output_text[index_number:].strip()

    # Calcular la precisión
    accuracy += 1 if output_text == number else 0

# Calcular la precisión
accuracy = accuracy / len(text)
print("Accuracy:", accuracy)

"""# Convertir la lista de arrays de NumPy en un solo array
activation_ffn_array = np.stack(activations)

# Redimensionar el array para obtener la forma deseada
activation_ffn_array = activation_ffn_array.reshape(num_layers, len(text), -1, num_neurons)

# save in pickle file
os.makedirs(folder, exist_ok=True)
with open(file, 'wb') as f:
    pickle.dump(activation_ffn_array, f)"""
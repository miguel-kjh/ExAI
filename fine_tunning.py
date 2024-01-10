from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import math
import random
import os

# deleted warnings
import warnings
warnings.filterwarnings("ignore")

# Cargar el modelo y el tokenizador
model_name = "gpt2"  # Puedes cambiar esto por otra variante de GPT-2 si lo prefieres
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def split_dataset(file_path, train_ratio=0.9):
    with open(file_path, encoding="utf-8") as file:
        lines = file.readlines()
        random.shuffle(lines)
        train_size = int(train_ratio * len(lines))
        return lines[:train_size], lines[train_size:]

# Preparar el conjunto de datos
file_path = os.path.join('datasets', 'dataset.txt')
train_lines, test_lines = split_dataset(file_path)

# Guardar los conjuntos de datos divididos
train_path = os.path.join('datasets', 'train_dataset.txt')
test_path = os.path.join('datasets', 'test_dataset.txt')
with open(train_path, "w", encoding="utf-8") as file:
    file.writelines(train_lines)
with open(test_path, "w", encoding="utf-8") as file:
    file.writelines(test_lines)

# Preparar los conjuntos de datos
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=128)
test_dataset = TextDataset(tokenizer=tokenizer, file_path=test_path, block_size=32)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",  # Directorio donde se guardarán los modelos
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=5,  # Ajusta esto según tus necesidades
    per_device_train_batch_size=4,  # Ajusta según los recursos de tu máquina
    save_steps=10_000,
    save_total_limit=2,
)

# Iniciar el entrenamiento
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Guardar el modelo afinado
trainer.save_model("./gpt2-finetuned")

# Realizar la evaluación después del entrenamiento y mostrar los resultados
# Evaluación para calcular la perplexidad
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

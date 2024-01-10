import os

def convert_line_to_numeric(line):
    # Separa el número y su representación textual
    number_str, text_representation = line.split(', ')
    # Elimina las comillas y los espacios adicionales
    text_representation = text_representation.strip().strip('"')
    return f"{text_representation}:{number_str}"

def convert_dataset(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as file:
        for line in lines:
            numeric_representation = convert_line_to_numeric(line)
            file.write(f"{numeric_representation}\n")

# Usar la función para convertir el dataset
file_path = os.path.join('datasets', 'dataset.txt')  
output_file = os.path.join('datasets', 'clean_dataset.txt')    # Asegúrate de cambiar esto por la ruta deseada para el archivo de salida

convert_dataset(file_path, output_file)

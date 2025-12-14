import onnxruntime as ort
import numpy as np
from PIL import Image
import ijson
import sys
import os
import glob

session = ort.InferenceSession("insect_model.onnx")

output_names = [output.name for output in session.get_outputs()]
output_shapes = [output.shape[1] for output in session.get_outputs()]

with open('hierarchy_map.json', 'rb') as f:
    ordres = []
    familles = []
    genres = []
    especes = []
    for key, value in ijson.kvitems(f, 'hierarchy_map'):
        ordres.append(value['ordre'])
        familles.append(value['famille'])
        genres.append(value['genre'])
        especes.append(value['espece'])

# Construire les listes de classes uniques sans utiliser set
ordre_classes = []
for item in ordres:
    if item not in ordre_classes:
        ordre_classes.append(item)
ordre_classes.sort()

famille_classes = []
for item in familles:
    if item not in famille_classes:
        famille_classes.append(item)
famille_classes.sort()

genre_classes = []
for item in genres:
    if item not in genre_classes:
        genre_classes.append(item)
genre_classes.sort()

espece_classes = []
for item in especes:
    if item not in espece_classes:
        espece_classes.append(item)
espece_classes.sort()

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Erreur : Le fichier {image_path} n'existe pas.")
        return

    print(f"Traitement de {image_path}:")

    image = Image.open(image_path).convert('RGB').resize((224, 224))

    # Convertir en numpy array et prétraiter
    image_array = np.array(image).astype(np.float32) / 255.0  # Scale to 0-1
    image_array = image_array.transpose(2, 0, 1)  # HWC to CHW

    # Normaliser avec les mêmes valeurs que ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std

    input_tensor = image_array[np.newaxis, ...]  # Add batch dim

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    outputs = session.run(output_names, {input_name: input_tensor})

    for name, output in zip(output_names, outputs):
        predicted = np.argmax(output, axis=1)[0]
        classes = globals()[f"{name}_classes"]
        if predicted < len(classes):
            print(f"  {name}: {classes[predicted]}")
        else:
            print(f"  {name}: Unknown")

if len(sys.argv) == 1:
    # Pas d'arg, traiter toutes les images du dossier
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(ext))
    if not image_paths:
        print("Aucune image trouvée dans le dossier.")
        sys.exit(1)
    for path in sorted(image_paths):
        process_image(path)
else:
    # Traiter l'image passée en arg
    process_image(sys.argv[1])

# Insect-ID-Script

Un script Python pour identifier les insectes à partir d'images en utilisant un modèle ONNX pré-entraîné. Le script classe les insectes aux niveaux taxonomiques suivants : ordre, famille, genre et espèce.

## Fonctionnalités

- Classification d'insectes à quatre niveaux taxonomiques
- Utilisation d'un modèle ONNX pour l'inférence rapide
- Prétraitement d'images avec Pillow et NumPy (léger, sans dépendances lourdes comme PyTorch)

## Prérequis

- Python 3.6+
- Les dépendances listées dans `requirements.txt`

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/YoannDev90/Insect-ID-Script.git
   cd Insect-ID-Script
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
   Ou utilisez le script automatique de configuration pour une installation optimisée selon votre matériel :
   ```bash
   ./setup.sh
   ```

## Utilisation

- Pour traiter une image spécifique :
  ```bash
  python main.py chemin/vers/votre/image.jpg
  ```

- Pour traiter toutes les images du dossier (jpg, jpeg, png, bmp) :
  ```bash
  python main.py
  ```

Le script affichera les prédictions pour chaque niveau taxonomique :
- ordre
- famille
- genre
- espèce

## Dépannage

### Installation des dépendances

- **Espace insuffisant dans /tmp** : L'installation de `onnxruntime` peut échouer si l'espace disque dans `/tmp` est insuffisant. Assurez-vous d'avoir au moins 10-20 Go d'espace libre dans `/tmp` ou définissez une variable d'environnement `TMPDIR` vers un répertoire avec plus d'espace :
  ```bash
  export TMPDIR=/path/to/large/tmp
  pip install -r requirements.txt
  ```

- **Détection automatique de l'OS et du matériel** : Pour optimiser les performances, installez la version appropriée d'ONNX Runtime en fonction de votre système :
  - Sur Linux avec GPU NVIDIA :
    ```bash
    pip install onnxruntime-gpu
    ```
  - Sur Linux avec GPU AMD ou Intel (ou CPU uniquement) :
    ```bash
    pip install onnxruntime
    ```
  - Sur Windows ou macOS, consultez la [documentation officielle d'ONNX Runtime](https://onnxruntime.ai/docs/install/) pour les versions GPU.

  Pour détecter automatiquement votre matériel, vous pouvez utiliser un script simple (non fourni) ou vérifier manuellement avec `nvidia-smi` pour NVIDIA ou `lspci | grep VGA` pour AMD/Intel.

### Utilisation du script

- **Ne faites pas confiance au modèle** : Ce modèle d'IA est pré-entraîné et peut produire des prédictions incorrectes. Utilisez les résultats comme une indication, pas comme une vérité absolue. Vérifiez toujours avec des experts ou des sources fiables.

- **Responsabilité** : Les auteurs de ce projet ne sont pas responsables des erreurs, dommages ou conséquences découlant de l'utilisation de ce script. Utilisez-le à vos propres risques.

- **Erreurs communes** :
  - Si l'image spécifiée n'existe pas, le script lèvera une erreur. Assurez-vous que le fichier est présent et accessible.
  - Pour les grandes images, assurez-vous que la mémoire RAM est suffisante.
  - Si le modèle ne charge pas, vérifiez que `insect_model.onnx` est dans le répertoire.

## Structure du projet

- `main.py` : Script principal pour l'identification des insectes
- `insect_model.onnx` : Modèle ONNX pré-entraîné
- `hierarchy_map.json` : Mappage des classes taxonomiques
- `requirements.txt` : Dépendances Python

## Dépendances

- onnxruntime : Pour l'exécution du modèle ONNX
- numpy : Pour les calculs numériques et le prétraitement d'images
- Pillow : Pour le chargement et le redimensionnement d'images
- ijson : Pour la lecture du fichier JSON hiérarchique

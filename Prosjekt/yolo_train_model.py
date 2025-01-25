from ultralytics import YOLO
import yaml
import os

# Filbaner til trenings- og valideringsdata
base_path = "C:/Users/andre/Desktop/AI-MASTER/2-SEMSTER/Computer Vision/Prosjekt/Dataset"
train_path = os.path.join(base_path, "train", "images").replace("\\", "/")
val_path = os.path.join(base_path, "val", "images").replace("\\", "/")

# Konfigurasjon av datasettet
dataset = {
    'train': train_path,
    'val': val_path,
    'nc': 37,  # Antall klasser
    'names': [f'class{i+1}' for i in range(37)]
}

# Lagre YAML-konfigurasjonen
yaml_file = 'dataset.yaml'
with open(yaml_file, 'w') as f:
    yaml.dump(dataset, f)

print(f"Dataset configuration saved to {yaml_file}")

# Last inn YOLO-modellen
model = YOLO("yolov8n.pt")

# Tren modellen
model.train(
    data=yaml_file,   # Sti til YAML-fil
    epochs=50,        # Antall epoker
    imgsz=640,        # Bildeoppløsning
    batch=8,          # Batch-størrelse
    optimizer="Adam", # Optimizer
    lr0=0.0001        # Læringsrate
)

# Lagre trenet modell
os.makedirs("yolo-Weights", exist_ok=True)  # Opprett mappe hvis den ikke finnes
model.save("yolo-Weights/custom_yolov8n.pt")

print("Trening fullført. Modell lagret som 'yolo-Weights/custom_yolov8n.pt'")

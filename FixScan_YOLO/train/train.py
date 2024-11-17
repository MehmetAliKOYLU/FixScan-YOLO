import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8n.pt")

# Eğitim fonksiyonu
def train_model(optimizer_name, lr, epochs=100):
    print(f"Training with optimizer: {optimizer_name}, learning rate: {lr}")
    
    results = model.train(
        data="./data.yaml",
        epochs=epochs,
        optimizer=optimizer_name,
        imgsz=640,
        lr0=lr,
        device=device,
        patience=10,
        workers=8
    )

if __name__ == '__main__':
    optimizer_name = 'Adam'#Kullandigim Optimizer isimleri
    learning_rate = 0.001#Kullandimi Learning rate değerleri

    # Modeli eğit ve sonuçları kaydet
    train_model(optimizer_name, learning_rate)

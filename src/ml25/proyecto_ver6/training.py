from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_loader
from network import Network

# Logging
import wandb
from datetime import datetime, timezone


def init_wandb(cfg):
    # Initialize wandb
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run = wandb.init(
        project="facial_expressions_cnn",
        config=cfg,
        name=f"facial_expressions_cnn_{timestamp}_utc",
    )
    return run


def validation_step(val_loader, net, cost_function):
    """
    Realiza un epoch completo en el conjunto de validación
    args:
    - val_loader (torch.DataLoader): dataloader para los datos de validación
    - net: instancia de red neuronal de clase Network
    - cost_function (torch.nn): Función de costo a utilizar

    returns:
    - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    """
    net.eval()  # Modo evaluación --agrega
    val_loss = 0.0
    correct = 0 #agrega
    total = 0 #agreg

    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch["transformed"]
        batch_labels = batch["label"]
        device = net.device
        batch_imgs = batch_imgs.to(device) #agrega
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            logits, proba = net(batch_imgs) #inicio cambios
            loss = cost_function(logits, batch_labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(proba.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item() #fin cambios
   
    # TODO: Regresa el costo promedio por minibatch
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    return avg_val_loss, val_acc


def train():
    # Hyperparametros
    cfg = {
        "training": {
            "learning_rate": 5e-4,  #cambio
            "n_epochs": 50,
            "batch_size": 128, #cambio
            "weight_decay": 1e-4, #agrega
            "scheduler_patience": 5, #agrega
            "scheduler_factor": 0.5,
            "early_stopping_patience": 15, #agrega
        },
    }
    run = init_wandb(cfg)

    train_cfg = cfg.get("training")
    learning_rate = train_cfg.get("learning_rate")
    n_epochs = train_cfg.get("n_epochs")
    batch_size = train_cfg.get("batch_size")
    weight_decay = train_cfg.get("weight_decay") #agrega

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader(
        "train", batch_size=batch_size, shuffle=True, num_workers=4 #num_workers por el gpu
        ) 
    val_dataset, val_loader = get_loader(
        "val", batch_size=batch_size, shuffle=False, num_workers=4 #num_workerss por el gpu
        )
    print(
        f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}"
    )

    # Instanciamos tu red
    modelo = Network(input_dim=48, n_classes=7)

    try:
        modelo.load_model("best_model.pth")
        print("continuando entrenamiento desde modelo anterior\n")
    except:
        print("entrenando desde cero\n")

    # CAMBIO PRINCIPAL: Aumentar peso de Disgusto de 1.5 a 5.0
    # Orden: [Enojo, Disgusto, Miedo, Alegria, Tristeza, Sorpresa, Neutral]
    class_weights = torch.tensor([
        1.0,   # 0: Enojo
        5.0,   # 1: Disgusto    ← CAMBIO: de 1.5 a 5.0 para compensar class imbalance
        1.2,   # 2: Miedo
        1.0,   # 3: Alegria
        1.0,   # 4: Tristeza
        1.0,   # 5: Sorpresa
        1.0    # 6: Neutral
    ]).to(modelo.device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\nPesos de clases aplicados:")
    print(f"  Enojo:    {class_weights[0]:.1f}")
    print(f"  Disgusto: {class_weights[1]:.1f}  ← AUMENTADO para compensar desbalance")
    print(f"  Miedo:    {class_weights[2]:.1f}")
    print(f"  Alegria:  {class_weights[3]:.1f}")
    print(f"  Tristeza: {class_weights[4]:.1f}")
    print(f"  Sorpresa: {class_weights[5]:.1f}")
    print(f"  Neutral:  {class_weights[6]:.1f}\n")

    # Define el optimizador
    optimizer = optim.Adam( #inicio cambios
        modelo.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=train_cfg.get("scheduler_factor"), 
        patience=train_cfg.get("scheduler_patience"),
        verbose=True
    )
    best_epoch_loss = np.inf
    patience_counter = 0
    early_stopping_patience = train_cfg.get("early_stopping_patience") #fin cambios

    best_epoch_loss = np.inf
    for epoch in range(n_epochs):
        modelo.train()  # Modo entrenamiento --agrega
        train_loss = 0.0
        correct = 0 #agrega
        total = 0 #agrega
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch["transformed"]
            batch_labels = batch["label"]
            
            batch_imgs = batch_imgs.to(modelo.device) #agrega (cuda, nvidia)
            batch_labels = batch_labels.to(modelo.device) #agrega (cuda, nvidia)

            # TODO Zero grad, forward pass, backward pass, optimizer step
            optimizer.zero_grad() #inicio cambios
            logits, proba = modelo(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step() #fin cambios

            # TODO acumula el costo
            train_loss += loss.item() #inicio cambios
            _, predicted = torch.max(proba.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item() #fin cambios

        # TODO Calcula el costo promedio
        train_loss = train_loss / len(train_loader) #cambio
        train_acc = 100 * correct / total #cambio
        
        val_loss, val_acc = validation_step(val_loader, modelo, criterion) #agrega val acc
        scheduler.step(val_loss) #agrega
        current_lr = optimizer.param_groups[0]['lr'] #agrega

        tqdm.write(
            f"epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}%, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%, lr: {current_lr:.6f}"  # --- agrega impresiones de accuracy y learning rate
        )

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        if val_loss < best_epoch_loss: #inicio cambios
            best_epoch_loss = val_loss
            modelo.save_model("best_model.pth")
            tqdm.write(f"mejor modelo guardado con val_loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            tqdm.write(f"early stopping activado después de {epoch} epochs")
            break #fin cambios

        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc, #agrega
                "val/loss": val_loss,
                "val/accuracy": val_acc, #agrega
                "learning_rate": current_lr, #agrega
            }
        )
    print(f"mejor val_loss alcanzado: {best_epoch_loss:.4f}")

if __name__ == "__main__":
    train()
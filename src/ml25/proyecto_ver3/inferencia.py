import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from  network import Network
import torch
from utils import (
    to_numpy,
    get_transforms,
    add_img_text,
)
from dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()


def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #lee directamente en escala de grises
    val_transforms, unnormalize = get_transforms("val", img_size=48) #cambiar test a val
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized


def predict(img_title_paths):
    """
    Hace la inferencia de las imagenes
    args:
    - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    """
    # Cargar el modelo
    modelo = Network(input_dim=48, n_classes=7)
    modelo.load_model("best_model.pth")
    modelo.eval() #linea agregada

    for module in modelo.modules(): #inicio cambio
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
            module.track_running_stats = False #fin cambio

    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        transformed = transformed.unsqueeze(0) #agregar dimension de batch

        # Inferencia
        logits, proba = modelo.predict(transformed) 
        proba_squeezed = proba.squeeze(0)
        pred = torch.argmax(proba.squeeze(0), -1).item() #squeeze para el eliminar dimension batch antes de argmax
        pred_label = EMOTIONS_MAP[pred]

        pred_confidence = proba.squeeze(0)[pred].item() * 100 #proba de prediccion

#muchos cambios (prueba 1)
        # Original / transformada
        h, w = original.shape[:2] if len(original.shape) > 1 else (original.shape[0], original.shape[0])
        resize_value = 300
        
        # Convertir a color para visualización si es necesario
        if len(original.shape) == 2:
            img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            img = original.copy()
            
        img = cv2.resize(img, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label} ({pred_confidence:.1f}%)")

        # Mostrar la imagen transformada
        denormalized = to_numpy(denormalized)
        if len(denormalized.shape) == 2 or denormalized.shape[2] == 1:
            denormalized_viz = cv2.cvtColor((denormalized * 255).astype(np.uint8).squeeze(), cv2.COLOR_GRAY2BGR)
        else:
            denormalized_viz = (denormalized * 255).astype(np.uint8)
        denormalized_viz = cv2.resize(denormalized_viz, (resize_value, resize_value))
        
        # Mostrar imágenes
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized_viz)
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC para salir
            break
    
    cv2.destroyAllWindows() #fin prueba 1


if __name__ == "__main__":
    # Direcciones relativas a este archivo
    test_folder = file_path / "test_imgs"
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    img_paths = []
    
    for file in test_folder.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            # Usar path relativo desde file_path
            img_paths.append(f"./test_imgs/{file.name}")

    predict(img_paths)
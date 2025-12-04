import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import (
    to_numpy,
    get_transforms,
    add_img_text,
)
from dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()


# NUEVA FUNCIÓN: Detectar y recortar rostro con Haar Cascade (MEJORADA + ECUALIZACIÓN)
def detect_and_crop_face(img):
    """
    Detecta y recorta el rostro usando Haar Cascade con múltiples intentos
    Incluye ecualización de histograma para mejorar detección con mala iluminación
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Ecualizar histograma para mejorar contraste (Opción A integrada)
    img_eq = cv2.equalizeHist(img)
    
    # Intento 1: Imagen ecualizada con parámetros estándar
    faces = face_cascade.detectMultiScale(img_eq, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Intento 2: Imagen original más sensible
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    
    # Intento 3: Imagen ecualizada muy sensible
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(img_eq, scaleFactor=1.03, minNeighbors=2, minSize=(15, 15))
    
    if len(faces) > 0:
        # Tomar el rostro más grande
        areas = [w * h for (x, y, w, h) in faces]
        largest_face_idx = np.argmax(areas)
        x, y, w, h = faces[largest_face_idx]
        
        # Recortar con padding del 20%
        padding = int(0.2 * w)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        face_cropped = img[y1:y2, x1:x2]
        return face_cropped, True, (x, y, w, h)
    else:
        return img, False, None


def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    
    # CAMBIO: Leer original a color para visualización
    img_original = cv2.imread(path)
    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)  # Convertir a gris
    
    # NUEVO: Detectar y recortar rostro
    img_face, face_detected, face_coords = detect_and_crop_face(img)
    
    if not face_detected:
        print("  ⚠️  No se detectó rostro, usando imagen completa")
    else:
        print("  ✓ Rostro detectado")
    
    val_transforms, unnormalize = get_transforms("val", img_size=48)
    tensor_img = val_transforms(img_face)  # CAMBIO: Usar img_face en lugar de img
    denormalized = unnormalize(tensor_img)
    
    # CAMBIO: Retornar también la original, face_detected y coords
    return img_original, img_face, tensor_img, denormalized, face_detected, face_coords


def predict(img_title_paths):
    """
    Hace la inferencia de las imagenes
    args:
    - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    """
    # Cargar el modelo
    modelo = Network(input_dim=48, n_classes=7)
    modelo.load_model("best_model.pth")
    modelo.eval()

    for module in modelo.modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
            module.track_running_stats = False

    for path in img_title_paths:
        print(f"\n{'='*50}")
        print(f"Procesando: {path}")
        print(f"{'='*50}")
        
        # Cargar la imagen
        im_file = (file_path / path).as_posix()
        # CAMBIO: Recibir más valores de load_img
        img_original, face_cropped, transformed, denormalized, face_detected, face_coords = load_img(im_file)

        transformed = transformed.unsqueeze(0)

        # Inferencia
        logits, proba = modelo.predict(transformed) 
        proba_squeezed = proba.squeeze(0)
        pred = torch.argmax(proba_squeezed, -1).item()
        pred_label = EMOTIONS_MAP[pred]
        pred_confidence = proba_squeezed[pred].item() * 100

        # NUEVO: Mostrar probabilidades
        print(f"\nProbabilidades:")
        for idx, (emo_idx, emo_name) in enumerate(EMOTIONS_MAP.items()):
            prob = proba_squeezed[emo_idx].item() * 100
            marker = " <<<" if idx == pred else ""
            print(f"  {emo_name:12s}: {prob:6.2f}%{marker}")
        print(f"\nPredicción: {pred_label} ({pred_confidence:.2f}%)")

        # Original / transformada
        # CAMBIO: Usar img_original y dibujar rectángulo si se detectó rostro
        img = img_original.copy()
        if face_detected and face_coords is not None:
            x, y, w, h = face_coords
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        h, w = img.shape[:2]
        resize_value = 300
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
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Direcciones relativas a este archivo
    test_folder = file_path / "test_imgs"
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    img_paths = []
    
    for file in sorted(test_folder.iterdir()):  # CAMBIO: sorted para orden
        if file.is_file() and file.suffix.lower() in valid_extensions:
            img_paths.append(f"./test_imgs/{file.name}")
    
    print(f"Se encontraron {len(img_paths)} imágenes")
    predict(img_paths)
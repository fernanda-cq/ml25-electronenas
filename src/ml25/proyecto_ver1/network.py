import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()

def build_backbone(model="resnet18", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "resnet18":
        backbone = resnet18(pretrained=weights == "imagenet")
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f"Model {model} not supported")


class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # TODO: Calcular dimension de salida
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
    
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
    
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        out_dim = 6*6*128

        # TODO: Define las capas de tu red
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(out_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
    
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, n_classes)  # Directo a las clases

        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = x.to(self.device) #inicio cambios
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
    
        x = x.view(x.size(0), -1)
    
        x = self.dropout1(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
    
        x = self.dropout2(x)
        logits = self.fc2(x)  # Sin fc3
        
        proba = F.softmax(logits, dim=1) #fin cambios
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        """
        Guarda el modelo en el path especificado
        args:
        - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
        - path (str): path relativo donde se guardará el modelo
        """
        models_path = file_path / "models" / model_name
        if not models_path.parent.exists():
            models_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path) #inicio cambios
        print(f"modelo guardado en: {models_path}") #fin cambios

    def load_model(self, model_name: str):
        """
        Carga el modelo en el path especificado
        args:
        - path (str): path relativo donde se guardó el modelo
        """
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / "models" / model_name #inicio cambios
        self.load_state_dict(torch.load(models_path, map_location=self.device, weights_only=True))
        print(f"modelo cargado desde: {models_path}") #fin cambios
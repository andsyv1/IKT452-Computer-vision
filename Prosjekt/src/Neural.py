import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = models.resnet18(weights=None)  # Bruker en liten versjon av ResNet
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Output = 10 tall (0-9)

    def forward(self, x):
        x = self.model(x)
        return x  # Ingen softmax nødvendig (CrossEntropyLoss håndterer det)

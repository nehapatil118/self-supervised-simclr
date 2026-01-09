import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        # Backbone encoder (ResNet-18)
        self.encoder = resnet18(weights=None)
        num_features = self.encoder.fc.in_features

        # Remove final classification layer
        self.encoder.fc = nn.Identity()

        # Projection head used in SimCLR
        self.projector = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)          # Encoder representation
        z = self.projector(h)        # Projected features
        z = F.normalize(z, dim=1)    # L2 normalization
        return z

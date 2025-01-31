import torch
import torch.nn as nn

class RewardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU()
        )
        self.reward_layer = nn.Sequential(
            nn.Linear(8192 + 32, 128),  
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, bbox):
        image_features = self.encoder(image)
        bbox_features = self.bbox_mlp(bbox)
        combined = torch.cat([image_features, bbox_features], dim=1)
        return self.reward_layer(combined)

import torch
import torch.nn as nn

class RewardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 32, 32)
            nn.ReLU(),
            nn.Flatten()  # Output: (128 * 32 * 32) = 131072
        )
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, 64),  # Bounding box features
            nn.ReLU()
        )
        self.reward_layer = nn.Sequential(
            nn.Linear(131072 + 64, 256),  # Combined features
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, image, bbox):
        image_features = self.encoder(image)  # Shape: (batch_size, 131072)
        bbox_features = self.bbox_mlp(bbox)  # Shape: (batch_size, 64)
        combined = torch.cat([image_features, bbox_features], dim=1)  # Shape: (batch_size, 131072 + 64)
        reward = self.reward_layer(combined)  # Shape: (batch_size, 1)
        return reward
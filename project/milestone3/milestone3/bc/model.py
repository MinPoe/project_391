import torch
import torch.nn as nn


class BCNet(nn.Module):
    """Behavioural Cloning network.

    Architecture (from project spec):
        181 LiDAR rays -> 256 (ReLU) -> 128 (ReLU) -> 2 (steering_angle, speed)
    """

    def __init__(self, num_lidar_rays: int = 181):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_lidar_rays, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

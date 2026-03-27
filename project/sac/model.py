import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACActorNet(nn.Module):
    """SAC Gaussian actor.

    Supports two modes:
        - BC-compatible (hidden2=128): initialised from BC weights, actions in [0, 1]
        - SB3-compatible (hidden2=256): converted from stable-baselines3, physical actions
    """

    def __init__(self, num_lidar_rays: int = 181, action_dim: int = 2,
                 hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(num_lidar_rays, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.mean_head = nn.Linear(hidden2, action_dim)
        self.log_std_head = nn.Linear(hidden2, action_dim)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.clamp(x_t, 0.0, 1.0)
        log_prob = dist.log_prob(x_t).sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                return mean
            std = log_std.exp()
            return Normal(mean, std).sample()

    @classmethod
    def from_bc(cls, bc_weights_path: str, num_lidar_rays: int = 181,
                action_dim: int = 2, device: str = "cpu"):
        """Create actor initialized from trained BC model weights."""
        actor = cls(num_lidar_rays, action_dim, hidden1=256, hidden2=128)
        bc_sd = torch.load(bc_weights_path, map_location=device, weights_only=True)
        actor.fc1.weight.data.copy_(bc_sd["net.0.weight"])
        actor.fc1.bias.data.copy_(bc_sd["net.0.bias"])
        actor.fc2.weight.data.copy_(bc_sd["net.2.weight"])
        actor.fc2.bias.data.copy_(bc_sd["net.2.bias"])
        actor.mean_head.weight.data.copy_(bc_sd["net.4.weight"])
        actor.mean_head.bias.data.copy_(bc_sd["net.4.bias"])
        nn.init.constant_(actor.log_std_head.weight, 0.0)
        nn.init.constant_(actor.log_std_head.bias, -2.0)
        return actor


class SACCriticNet(nn.Module):
    """SAC Q-function: (state, action) -> scalar Q-value."""

    def __init__(self, num_lidar_rays: int = 181, action_dim: int = 2,
                 hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_lidar_rays + action_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))

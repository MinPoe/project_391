import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SACActorNet(nn.Module):
    """SAC Gaussian actor with tanh squashing.

    Same hidden architecture as BCNet (181 -> 256 -> 128) with separate
    mean and log_std output heads.  Outputs actions in [-1, 1] via tanh.
    """

    def __init__(self, num_lidar_rays: int = 181, action_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(num_lidar_rays, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mean_head = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """Sample action with reparameterization trick + tanh squashing.

        Returns:
            action  – tanh-squashed action in [-1, 1]
            log_prob – corrected log-probability (scalar per sample)
            mean    – raw mean (useful for logging)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()                       # reparameterized sample
        action = torch.tanh(x_t)
        # log-prob with Jacobian correction for tanh squashing
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Get action for driving (no gradient tracking)."""
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                return torch.tanh(mean)
            std = log_std.exp()
            return torch.tanh(Normal(mean, std).sample())

    @classmethod
    def from_bc(cls, bc_weights_path: str, num_lidar_rays: int = 181,
                action_dim: int = 2, device: str = "cpu"):
        """Create actor initialized from trained BC model weights.

        Hidden layers (LiDAR feature extraction) are copied exactly.
        The mean head is seeded from the BC output layer.
        The log_std head starts with small values (conservative exploration).
        """
        actor = cls(num_lidar_rays, action_dim)
        bc_sd = torch.load(bc_weights_path, map_location=device, weights_only=True)
        actor.fc1.weight.data.copy_(bc_sd["net.0.weight"])
        actor.fc1.bias.data.copy_(bc_sd["net.0.bias"])
        actor.fc2.weight.data.copy_(bc_sd["net.2.weight"])
        actor.fc2.bias.data.copy_(bc_sd["net.2.bias"])
        actor.mean_head.weight.data.copy_(bc_sd["net.4.weight"])
        actor.mean_head.bias.data.copy_(bc_sd["net.4.bias"])
        nn.init.constant_(actor.log_std_head.weight, 0.0)
        nn.init.constant_(actor.log_std_head.bias, -1.0)   # std ~ 0.37
        return actor


class SACCriticNet(nn.Module):
    """SAC Q-function: (state, action) -> scalar Q-value."""

    def __init__(self, num_lidar_rays: int = 181, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_lidar_rays + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))

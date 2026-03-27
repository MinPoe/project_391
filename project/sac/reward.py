import numpy as np


def compute_reward(
    lidar_ranges: np.ndarray,
    speed: float,
    steering_angle: float,
    done: bool,
) -> float:
    """Compute single-step reward for SAC training.

    Args:
        lidar_ranges: Downsampled LiDAR distances in **meters** (before
            normalization), shape (num_rays,).
        speed: Physical forward speed in m/s.
        steering_angle: Physical steering angle in radians.
        done: True when the episode ends (emergency stop / crash).

    Returns:
        Scalar reward value.
    """
    reward = 0.0

    # 1. Forward progress -- encourage speed
    reward += speed * 1.0

    # 2. Wall proximity -- penalise being close to obstacles
    min_range = float(np.min(lidar_ranges))
    if min_range < 0.2:
        reward -= (0.2 - min_range) * 5.0

    # 3. Track centering -- balanced left / right distances
    n = len(lidar_ranges)
    left_avg = float(np.mean(lidar_ranges[: n // 4]))
    right_avg = float(np.mean(lidar_ranges[3 * n // 4 :]))
    reward -= abs(left_avg - right_avg) * 0.1

    # 4. Steering smoothness
    reward -= abs(steering_angle) * 0.3

    # 5. Crash / emergency-stop penalty
    if done:
        reward -= 10.0

    return reward

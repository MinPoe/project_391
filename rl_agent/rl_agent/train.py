import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from rl_agent.rl_env import F1TenthEnv

MODEL_DIR = os.path.expanduser('~/rl_models')
LOG_DIR = os.path.expanduser('~/rl_logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class TrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0
        self.episode_count = 0
        self.current_max_speed = 2.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals['rewards'][0]
        self.current_length += 1

        # Curriculum: increase speed as training progresses
        if self.num_timesteps >= 5000 and self.current_max_speed < 2.5:
            self.current_max_speed = 2.5
            self.training_env.env_method('set_max_speed', 2.5)
            print('\n --> Curriculum: max speed increased to 2.5 m/s')
        elif self.num_timesteps >= 15000 and self.current_max_speed < 3.0:
            self.current_max_speed = 3.0
            self.training_env.env_method('set_max_speed', 3.0)
            print('\n --> Curriculum: max speed increased to 3.0 m/s')

        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)

            avg_reward = np.mean(self.episode_rewards[-10:])
            print(f"Episode {self.episode_count:4d} | "
                  f"Reward: {self.current_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Length: {self.current_length:5d} steps | "
                  f"Speed: {self.current_max_speed:.1f}")

            self.current_reward = 0
            self.current_length = 0

            if self.episode_count % 10 == 0:
                path = os.path.join(MODEL_DIR, f'sac_checkpoint_{self.episode_count}')
                self.model.save(path)
                print(f"  --> Checkpoint saved to {path}")

        return True


def main():
    env = F1TenthEnv()
    env = TimeLimit(env, max_episode_steps=1000)

    checkpoint_path = os.path.expanduser('~/rl_models/sac_final.zip')
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = SAC.load(checkpoint_path, env=env, device='cpu')
    else:
        print("No checkpoint found, starting fresh...")
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            learning_starts=500,
            verbose=0,
            tensorboard_log=LOG_DIR,
            device='cpu',
        )

    print("Starting training. Press Ctrl+C to stop.\n")

    try:
        model.learn(
            total_timesteps=200_000,
            callback=TrainingCallback(),
            progress_bar=False,
            reset_num_timesteps=False,  # keeps timestep count when resuming
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    final_path = os.path.join(MODEL_DIR, 'sac_final')
    model.save(final_path)
    print(f"Model saved to {final_path}")
    env.close()


if __name__ == '__main__':
    main()
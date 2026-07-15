import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FpsEnv(gym.Env):
    """
    A generic FPS Reinforcement Learning environment.
    This serves as an abstract simulation for research purposes.
    Updated to be compatible with stable-baselines3 (flattened spaces).
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Define action space:
        # Flattened: [move_x, move_y, look_x, look_y, shoot_probability]
        # All continuous values between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # Define observation space:
        # A generic representation: 
        # For simplicity with standard CNN policies, we will use just the "vision" matrix.
        # In a real scenario, you'd use MultiInputPolicy for Dict spaces, but Dict actions are unsupported.
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Initialize generic state
        observation = np.zeros((64, 64, 3), dtype=np.uint8)
        info = {}
        return observation, info

    def step(self, action):
        # Simulate environment step
        self.step_count += 1
        
        # Dummy observation update
        observation = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Dummy reward
        reward = 0.1
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            pass # Placeholder for actual rendering

    def close(self):
        pass

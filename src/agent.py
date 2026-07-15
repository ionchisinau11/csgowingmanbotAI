import os
from stable_baselines3 import PPO
from src.fps_env import FpsEnv

def main():
    print("Initializing environment...")
    env = FpsEnv()
    
    print("Initializing PPO agent...")
    # Use CnnPolicy since our observation is an image
    model = PPO("CnnPolicy", env, verbose=1)
    
    print("Starting training for 500 timesteps...")
    model.learn(total_timesteps=500)
    
    print("Training finished. Saving model to 'ppo_fps_agent.zip'...")
    model.save("ppo_fps_agent")
    
    print("Evaluating trained agent...")
    obs, info = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action taken by model: {action}")
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()

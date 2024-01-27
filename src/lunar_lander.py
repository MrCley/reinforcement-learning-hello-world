"""
    Simple lunar lander environment with basic physics.
    Saving the model for continuous training
"""
import gymnasium as gym
from stable_baselines3 import DQN, PPO

# render_mode = "human" or "rgb_array"
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

# load or create model
try:
    # model = DQN.load("./models/dqn_lunar_lander", env, verbose=1)
    model = PPO.load("./models/lunar_lander/ppo_lunar_lander", env, verbose=1)
except Exception as e:
    # model = DQN("MlpPolicy", env, verbose=1, buffer_size=50_000, batch_size=32)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_lunar_lander")

# training 
epochs = 100
timesteps = 500_000 

for i in range(epochs):
    try:
        model.learn(total_timesteps=timesteps, log_interval=100, reset_num_timesteps=False, progress_bar=True)
        model.save("./models/ppo_lunar_lander")
    except Exception as e:
        print(e)

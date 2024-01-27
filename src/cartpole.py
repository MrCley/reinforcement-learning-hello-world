"""
    What is reinforcement learning?
    Reinforcement learning is a type of machine learning that enables an agent to learn in an interactive environment
    by trial and error using feedback from its own actions and experiences.

    Reinforcement learning falls under the umbrella of machine learning, but it differs from supervised learning
    because we don't need to supervise (or provide the expected value) the model's training.  

    This is a "hello world" example of reinforcement learning using the classic CartPole environment:
    A cart is attached to a pole, and the goal is to prevent it from falling over by applying a force that moves the cart left or right.
    
"""

import gymnasium as gym
from stable_baselines3 import PPO 

"""
    Create Environment:
    In Reinforcement Learning, the environment is the one that provides the agent with the state and reward 
    while the agent provides the environment with the action. 

    Environment loop in Reinforcement Learning (State, Reward and Action):
        1. State is the current situation of the agent (The current frame of a visual environment). 
        2. Action is the decision made by the agent based on the state.
        3. Reward is the feedback from the environment based on the agent action.

    The idea is to maximize the reward by taking the best action in each state.
"""

# This trains the model in rgb mode (much faster) then displays the results in human mode
# change the render_mode to "human" to also vizualize the training process
env = gym.make("CartPole-v1", render_mode="rgb_array")
 

"""
    Create PPO Model:
    PPO (Proximal Policy Optimization) is a reinforcement learning algorithm published by OpenAI in 2017:
    https://arxiv.org/abs/1707.06347 and is ofter used for continuous action spaces (like the CartPole environment).
"""
ppo = PPO("MlpPolicy", env, verbose=1)
total_timesteps = 500_000
ppo.learn(total_timesteps=total_timesteps, log_interval=1000, progress_bar=True) 


"""
    Evaluate DQN Model:
    The model will play the environment using the learned optimal policy.
    If the pole is not falling, the agent is doing a good job.
""" 
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for i in range(total_timesteps):
    action, _ = ppo.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
"""
    Play Atari saved models
"""
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
 
model_file = "./models/enduro/dqn_enduro"
env = make_atari_env("EnduroNoFrameskip-v4", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

model = DQN.load(model_file, env, verbose=1)
obs = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action) 
    
    env.render(mode='human')

    if dones.any(): 
        obs = env.reset()

"""
    Since the "Atari paper" (https://arxiv.org/abs/1312.5602) came out, Atari games have become a popular RL benchmark 
    due to their relative complex tasks but lightweight computational requirements.

    This paper also made the case for using Convolutional Neural Networks (CNN) for RL tasks and popularized the DQN
    (Deep Q Network) algorithm used in this example.

    But why is it so hard to train an agent to play Atari games?
    The main challenge is that the environment is visual and the agent cannot access the internal state of the game, 
    so the agent has to learn from the screen pixels how to play the game.

    Note: training takes a while: the original paper trained their DQN for 10 million frames. (a few hours on decent hardware)
"""
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack 

"""
    SpaceInvadersNoFrameskip-v4
    MsPacmanNoFrameskip-v4
    BreakoutNoFrameskip-v4
    EnduroNoFrameskip-v4
    KungFuMasterNoFrameskip-v4
"""
# Set the environment/file name
env_name = "BreakoutNoFrameskip-v4"   
model_file = "./models/dqn_breakout"
tensorboard_log = f"./logs/dqn_breakout"

"""
    SBL3 wraper: vectorizes the environment to optimize the training, 
    this enables multiple envs (n_envs) to run in parallel
    Stack wraper: Stacks 4 frames to add motion information
"""
env = make_atari_env(env_name, n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

"""
    Load or create the DQN model, the default hyperparameters values can be checked in the DQN constructor
    they are good enough for Atari games, but I'd reduce the learning rate once the model starts to converge (stops learning)
    and increase the buffer/batch sizes depending on your hardware
"""
try:
    model = DQN.load(model_file, env, verbose=1)
    # model = PPO.load(model_file, env, verbose=1)

    print("\nModel loaded ᕕ(⌐■_■)ᕗ ♪♬\n")
except Exception as e:
    model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log, learning_rate=1e-4, buffer_size=100_000, batch_size=32)
    # model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log)

    print(f'\nNo model found for {env_name}, starting from scratch! (╯°□°）╯ ︵ ┻━┻\n')
    print(e)
finally:
    print("Starting training ᕙ(⇀‸↼‶)ᕗ\n")


# train the model during epochs x timesteps frames
epochs = 500
timesteps = 100_000 

for i in range(epochs):
    try:
        model.learn(total_timesteps=timesteps, log_interval=100, reset_num_timesteps=False, progress_bar=True)
        model.save(model_file)
    except Exception as e:
        print(e)
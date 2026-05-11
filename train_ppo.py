from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from traffic_env import TrafficEnv

import os


os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "logs",
    exist_ok=True
)

# Parallel environments
env = make_vec_env(

    TrafficEnv,

    n_envs=16
)

model = PPO(

    "MlpPolicy",

    env,

    verbose=1,

    tensorboard_log="./logs/",

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=256,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    device="cpu"
)

model.learn(
    total_timesteps=500000
)

model.save(
    "models/ppo_traffic"
)

print("\nTraining Complete!")
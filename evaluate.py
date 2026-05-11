from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from traffic_env import TrafficEnv


env = TrafficEnv()

model = PPO.load(
    "models/ppo_traffic"
)

mean_reward, std_reward = evaluate_policy(

    model,

    env,

    n_eval_episodes=20
)

print(
    f"Mean Reward: {mean_reward:.2f}"
)

print(
    f"Std Reward: {std_reward:.2f}"
)
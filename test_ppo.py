from stable_baselines3 import PPO

from traffic_env import TrafficEnv

import time


env = TrafficEnv()

model = PPO.load(
    "models/ppo_traffic"
)

obs, _ = env.reset()

done = False

total_reward = 0

step = 0

while not done:

    action, _ = model.predict(

        obs,

        deterministic=True
    )

    obs, reward, done, _, _ = env.step(action)

    total_reward += reward

    print("\n===================")

    print(f"Step: {step}")

    print(
        f"Cars [N,S,E,W]: {env.cars}"
    )

    print(

        "Signal:",

        "NS Green"
        if action == 0
        else "EW Green"
    )

    print(

        f"Wait Times -> "
        f"NS: {env.ns_wait} | "
        f"EW: {env.ew_wait}"
    )

    print(
        f"Reward: {reward:.2f}"
    )

    step += 1

    time.sleep(0.15)

print("\n===================")

print(
    "Final Reward:",
    total_reward
)
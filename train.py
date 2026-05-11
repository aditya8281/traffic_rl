from traffic_env import TrafficEnv
from dqn_agent import DQNAgent

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Environment
env = TrafficEnv()

# Agent
state_size = 5
action_size = 2

agent = DQNAgent(
    state_size,
    action_size
)

# Training settings
episodes = 300

# Store rewards
episode_rewards = []

for episode in range(episodes):

    state, _ = env.reset()

    total_reward = 0

    done = False

    while not done:

        # Choose action
        action = agent.act(state)

        # Take action
        next_state, reward, done, _, _ = env.step(action)

        # Store experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Learn from replay
        agent.replay()

        # Update state
        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    print(
        f"Episode: {episode+1}/{episodes} | "
        f"Reward: {total_reward:.2f} | "
        f"Epsilon: {agent.epsilon:.4f}"
    )

# Save trained model
torch.save(
    agent.model.state_dict(),
    "models/dqn_traffic.pth"
)

print("\nModel saved!")

# Plot rewards
plt.figure(figsize=(10, 5))

plt.plot(episode_rewards)

plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.title("DQN Traffic RL Training")

plt.grid(True)

plt.savefig("plots/training_rewards.png")

plt.show()
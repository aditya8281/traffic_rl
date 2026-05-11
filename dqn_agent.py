import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Neural Network
class DQN(nn.Module):

    def __init__(self, state_size, action_size):

        super(DQN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)

        )

    def forward(self, x):

        return self.network(x)


# DQN Agent
class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # Replay memory
        self.memory = deque(maxlen=5000)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.batch_size = 64

        self.learning_rate = 0.001

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # Main network
        self.model = DQN(
            state_size,
            action_size
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def act(self, state):

        # Exploration
        if random.random() < self.epsilon:

            return random.randint(
                0,
                self.action_size - 1
            )

        # Exploitation
        state = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        q_values = self.model(state)

        return torch.argmax(
            q_values
        ).item()

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in batch:

            state = torch.FloatTensor(
                state
            ).to(self.device)

            next_state = torch.FloatTensor(
                next_state
            ).to(self.device)

            target_q = reward

            if not done:

                target_q += self.gamma * torch.max(
                    self.model(next_state)
                ).item()

            current_q = self.model(
                state
            )[action]

            target_q = torch.tensor(
                target_q,
                dtype=torch.float32
            ).to(self.device)

            loss = self.loss_fn(
                current_q,
                target_q
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay
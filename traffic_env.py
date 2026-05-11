import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficEnv(gym.Env):

    def __init__(self):

        super(TrafficEnv, self).__init__()

        # Actions:
        # 0 = North-South green
        # 1 = East-West green
        self.action_space = spaces.Discrete(2)

        # State:
        # [north, south, east, west, current_signal]
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(5,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Random cars at start
        self.cars = np.random.randint(
            0,
            20,
            size=4
        )

        # Current green signal
        self.current_green = 0

        # Step counter
        self.steps = 0

        return self._get_state(), {}

    def _get_state(self):

        return np.array([
            self.cars[0],  # north
            self.cars[1],  # south
            self.cars[2],  # east
            self.cars[3],  # west
            self.current_green
        ], dtype=np.float32)

    def step(self, action):

        self.current_green = action

        # Cars pass depending on signal
        if action == 0:

            # North-South green
            self.cars[0] = max(0, self.cars[0] - 5)
            self.cars[1] = max(0, self.cars[1] - 5)

        else:

            # East-West green
            self.cars[2] = max(0, self.cars[2] - 5)
            self.cars[3] = max(0, self.cars[3] - 5)

        # New incoming cars
        self.cars += np.random.randint(
            0,
            4,
            size=4
        )

        # Total waiting cars
        waiting_time = np.sum(self.cars)

        # Reward
        reward = -waiting_time

        self.steps += 1

        # Episode ends after fixed steps
        done = self.steps >= 100

        return (
            self._get_state(),
            reward,
            done,
            False,
            {}
        )
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficEnv(gym.Env):

    def __init__(self):

        super(TrafficEnv, self).__init__()

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=0,
            high=500,
            shape=(7,),
            dtype=np.float32
        )

        self.max_steps = 200

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.cars = np.random.randint(
            0,
            10,
            size=4
        )

        self.current_green = 0

        self.steps = 0

        self.ns_wait = 0
        self.ew_wait = 0

        return self._get_state(), {}

    def _get_state(self):

        return np.array([

            self.cars[0],
            self.cars[1],
            self.cars[2],
            self.cars[3],

            self.current_green,

            self.ns_wait,
            self.ew_wait

        ], dtype=np.float32)

    def step(self, action):

        switch_penalty = 0

        if action != self.current_green:
            switch_penalty = 1

        self.current_green = action

        # NS Green
        if action == 0:

            passed = (
                min(5, self.cars[0])
                + min(5, self.cars[1])
            )

            self.cars[0] = max(
                0,
                self.cars[0] - 5
            )

            self.cars[1] = max(
                0,
                self.cars[1] - 5
            )

            self.ns_wait = 0

            self.ew_wait += (
                self.cars[2]
                + self.cars[3]
            )

        # EW Green
        else:

            passed = (
                min(5, self.cars[2])
                + min(5, self.cars[3])
            )

            self.cars[2] = max(
                0,
                self.cars[2] - 5
            )

            self.cars[3] = max(
                0,
                self.cars[3] - 5
            )

            self.ew_wait = 0

            self.ns_wait += (
                self.cars[0]
                + self.cars[1]
            )

        # Dynamic traffic generation
        incoming = np.random.poisson(
            lam=np.random.uniform(0.5, 2.0),
            size=4
        )

        self.cars += incoming

        waiting_time = np.sum(self.cars)

        total_wait = (
            self.ns_wait
            + self.ew_wait
        )

        # Reward shaping
        reward = (
            passed * 2
            - 0.2 * waiting_time
            - 0.01 * total_wait
            - 0.5 * switch_penalty
        )

        if waiting_time < 10:
            reward += 5

        if waiting_time > 60:
            reward -= 10

        self.steps += 1

        terminated = (
            self.steps >= self.max_steps
        )

        return (
            self._get_state(),
            reward,
            terminated,
            False,
            {}
        )
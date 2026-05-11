from traffic_env import TrafficEnv

env = TrafficEnv()

state, _ = env.reset()

print("Initial State:", state)

for step in range(10):

    action = env.action_space.sample()

    next_state, reward, done, _, _ = env.step(action)

    print(f"\nStep {step+1}")

    print("Action:", action)

    print("Next State:", next_state)

    print("Reward:", reward)
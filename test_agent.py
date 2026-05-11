from dqn_agent import DQNAgent

agent = DQNAgent(
    state_size=5,
    action_size=2
)

print(agent.model)
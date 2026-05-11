# Adaptive Traffic Signal Control using Reinforcement Learning

A Reinforcement Learning based adaptive traffic signal optimization system built using **PPO (Proximal Policy Optimization)** and a custom **Gymnasium** environment.

The agent learns to dynamically control traffic signals to reduce congestion, waiting time, and inefficient signal switching.

---

# Features

- PPO Reinforcement Learning
- Custom Gymnasium Environment
- Vectorized Parallel Training
- Adaptive Traffic Signal Optimization
- Reward Engineering
- Waiting Time Memory
- Dynamic Traffic Generation
- TensorBoard Logging
- Evaluation Pipeline
- Stable-Baselines3 Integration

---

# Project Structure

```text
traffic_rl/
│
├── traffic_env.py
├── train_ppo.py
├── test_ppo.py
├── evaluate.py
├── requirements.txt
├── .gitignore
│
├── models/
├── logs/
```

---

# Reinforcement Learning Overview

The traffic signal agent observes the environment state:

```text
[
    North Cars,
    South Cars,
    East Cars,
    West Cars,
    Current Green Signal,
    North-South Wait Time,
    East-West Wait Time
]
```

The agent then selects one of two actions:

```text
0 → North-South Green
1 → East-West Green
```

The PPO agent learns a policy:

```math
\pi(a|s)
```

that maximizes long-term traffic flow efficiency.

---

# Reward Design

The reward function encourages:

- reducing congestion
- minimizing total waiting time
- avoiding excessive signal switching
- maximizing traffic throughput

The agent receives:
- positive rewards for clearing vehicles
- penalties for congestion and long waits

---

# Technologies Used

- Python
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- TensorBoard

---

# Installation

Clone the repository:

```bash
git clone https://github.com/aditya8281/traffic_rl.git
cd traffic_rl
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

## Windows

```bash
.venv\Scripts\activate
```

## Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Train the PPO agent:

```bash
python train_ppo.py
```

The trained model will be saved in:

```text
models/
```

---

# Testing

Run the trained agent:

```bash
python test_ppo.py
```

The simulation will display:
- traffic conditions
- selected signal
- waiting times
- rewards

---

# Evaluation

Evaluate average policy performance:

```bash
python evaluate.py
```

---

# TensorBoard Visualization

Launch TensorBoard:

```bash
tensorboard --logdir logs
```

Open in browser:

```text
http://localhost:6006
```

You can monitor:
- episode rewards
- policy loss
- entropy
- training stability

---

# RL Concepts Used

- Policy Gradient Methods
- PPO (Proximal Policy Optimization)
- Generalized Advantage Estimation
- Reward Engineering
- Vectorized Environments
- Stochastic Environment Simulation

---

# Future Improvements

- Multi-intersection traffic networks
- SUMO traffic simulator integration
- Multi-agent reinforcement learning
- Graph Neural Networks
- Real traffic datasets
- Camera-based traffic input
- Transformer-based RL policies

---

# Results

The PPO agent learns adaptive traffic control strategies that outperform random signal switching and significantly reduce congestion over training.

---

# Author

Aditya

```

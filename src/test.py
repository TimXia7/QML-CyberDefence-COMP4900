import pennylane as qml
import numpy as np
import torch
import random

# Quantum device with 1 qubit
dev = qml.device("default.qubit", wires=1)

# Variational quantum circuit (VQC) to approximate Q-values
@qml.qnode(dev, interface="torch")
def quantum_q_network(theta_x, theta_y):
    qml.RX(theta_x, wires=0)
    qml.RY(theta_y, wires=0)
    return qml.expval(qml.PauliZ(0))  # This represents Q(s, a)

# Initialize Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Trainable parameters (θ_x, θ_y)
theta_x = torch.tensor(1.0, requires_grad=True)  # Initialize with some value
theta_y = torch.tensor(1.0, requires_grad=True)

optimizer = torch.optim.Adam([theta_x, theta_y], lr=alpha)

# Define a Q-table (simplified for a small problem)
Q_table = np.zeros((5, 2))  # Example with 5 states and 2 actions

# Reward function (example: output close to 1 gives high reward)
def get_reward(theta_x, theta_y):
    output = quantum_q_network(theta_x, theta_y)
    target = 1.0  # We want the circuit to output close to 1
    return -(output - target) ** 2  # Negative MSE as the reward

# Q-learning loop
for episode in range(100):
    # Choose state randomly
    state = random.randint(0, 4)
    
    # Choose action (randomly or best action from Q-table)
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, 1)  # Exploration
    else:
        action = np.argmax(Q_table[state])  # Exploitation

    # Compute reward using the quantum circuit
    reward = get_reward(theta_x, theta_y)
    
    # Update Q-value using the Q-learning equation
    Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[state]) - Q_table[state, action])
    
    # Optimize θ parameters using gradient descent
    optimizer.zero_grad()
    loss = -reward  # Maximize reward
    loss.backward()
    optimizer.step()

    print(f"Episode {episode}: θ_x = {theta_x.item()}, θ_y = {theta_y.item()}, Reward = {reward}")

print("Final Q-table:", Q_table)

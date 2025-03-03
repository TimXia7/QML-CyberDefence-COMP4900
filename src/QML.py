
# This file holds the ML components and related computations

# Tim: To my understanding, we want to set all of our initial ML values here (from Q-learning) such as learning rate (alpha)
# Define the reward function here which uses function, variational_circuit. And adjust theta_x and theta_y, and loop etc.

import pennylane as qml
import numpy as np
from variational_circuit import variational_circuit

# Initial Q-Learning ML values here?
# ...

# Example initial values, both pi/4
theta_x = np.pi / 4
theta_y = 0.78539816339

# Reward function here to determine how effective the agent was at avoiding the adversary?
# def train_avoidant_simulation(theta_x, theta_y):
# ... 

# Q-Learning loop here?
# ...




# Test output for variational circuit
print(variational_circuit(theta_x, theta_y))




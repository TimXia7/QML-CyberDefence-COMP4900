import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import random

from train_simulation import *


# QML code, loosely based off of the paper, extra materials, and the given mlx file
dev = qml.device("default.qubit", wires=1, shots=1000)

# Basic VQC
@qml.qnode(dev)
def W(theta):
    qml.RX(theta[0], wires=0)
    qml.RY(theta[1], wires=0)
    return qml.probs(wires=[0])

# Return mean squared error (MSE)
# Purpose: determine how desireable the output is. Smaller, the better
def square_loss(X, Y):
    loss = 0
    for x, y in zip(X, Y):
        loss = loss + (x - y) ** 2
    return loss / len(X)

# Evaluate the QC based on the current probabilty distribution
def cost(theta, p):
    return square_loss(W(theta), [p, 1-p])

# Value for how fast the system leanrs (policy change speed)
opt = qml.GradientDescentOptimizer(stepsize=0.01)

# How many times the optimizer will try to improve the quantum model
# see update() below:
steps = 100  

# Trains quantum circuit to the expected probabity
# More specifically, this is what adjusts the theta value
def update(theta, p):
    for i in range(steps):
        theta = opt.step(lambda v: cost(v, p), theta)
    return theta

# Q-Learning values
theta = np.random.randn(2)  # Initialize circuit parameters randomly
epochs = 100               # Training iterations (see main loop below)
M = np.zeros(epochs)        # Stores probability of ket 0. This is used later to graph everything
Q = [0, 0]                  # Q-values for actions (loop, bypass)
alpha = 0.1                # Learning rate (similar to stepsize, but it is Q-Learning specific)
gamma = 0.95                # Discount factor for future rewards
epsilon = 0.1

# Main Training Loop
track = Track()
train1 = Train(track, start_position=0)
train2 = Train(track, start_position=7)
distances = np.zeros(epochs)

previous_distance = simulate_train_loop(train1, train2, track, np.random.randint(2))

for i in range(epochs):
    if np.random.rand() < epsilon:
        a = np.random.randint(2)
    else:
        a = np.argmax(Q)

    take_bypass = a == 1

    # Determine reward (simulate_train_loop returns the distance between the trains)
    current_distance = simulate_train_loop(train1, train2, track, take_bypass)
    distances[i] = current_distance

    if current_distance > previous_distance:
        reward = +1
    elif current_distance < previous_distance:
        reward = -1
    else:
        reward = 0 

    # Update Q-values using Bellman equation (I learned this from the prof's code)
    # to my knowledge, this is just part of the Q-learning process
    # Not in the code in twotrainexample.ipynb though, so I'm looking more into this
    Q[a] = Q[a] + alpha * (reward + gamma * max(Q) - Q[a])

    # Train the system based on Q-values using update()
    # Similar to prof's code
    if abs(Q[0]) + abs(Q[1]) > 0:
        theta = update(theta, Q[0] / (Q[0] + Q[1]))

    # Store probability of choosing loop to graph later
    M[i] = W(theta)[0]
    

# Plot results
plt.plot(M, 'bs', label='Take Loop')
plt.plot(1 - M, 'g^', label='Take Bypass')
plt.xlabel('Epoch')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('plot.png')  # Save as PNG


plt.figure()
plt.plot(distances, 'ro-', label='Distance Between Trains')

window = 10
avg_distances = np.convolve(distances, np.ones(window)/window, mode='valid')
plt.plot(range(window - 1, epochs), avg_distances, 'b-', label=f'{window}-Epoch Average')

plt.xlabel('Epoch')
plt.ylabel('Distance')
plt.title('Distance Between Trains Over Time')
plt.grid(True)
plt.legend()
plt.savefig('distance_plot.png')
plt.show()

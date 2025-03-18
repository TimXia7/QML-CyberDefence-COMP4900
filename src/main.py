import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

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

alpha_values = [0.1]
gamma_values = [0.9]
epsilon_values = [0.1]
epoch_values = [100]

results = []

mode = "QRL"

# Sweep through all combinations of parameters
for alpha, gamma, epsilon, epochs in itertools.product(alpha_values, gamma_values, epsilon_values, epoch_values):
    print(f"Running with alpha={alpha}, gamma={gamma}, epsilon={epsilon}, epochs={epochs}")

    # Initialize Q-Learning values
    theta_train1 = np.random.randn(2)
    theta_train2 = np.random.randn(2)
    M = np.zeros(epochs)  # Probability of taking the loop
    A = np.zeros(epochs)
    train1_Q = [0, 0]
    train2_Q = [0, 0]
    distances = np.zeros(epochs)

    # Initialize the track and trains
    track = AdvancedTrack()
    train1 = Train(track, start_position=0)
    train2 = Train(track, start_position=7)

    previous_distance = calculate_distance(train1, train2, track)

    for i in range(epochs):
        if np.random.rand() < epsilon:
            a_train1 = np.random.randint(2)
        else:
            a_train1 = np.argmax(train1_Q)
        
        take_bypass_train1 = a_train1 == 1

        

        # Simulate train movement and get the reward
        if mode == "QRL":
            if np.random.rand() < epsilon:
                a_train2 = np.random.randint(2)
            else:
                a_train2 = np.argmax(train2_Q)
            
            take_bypass_train2 = a_train2 == 1
            current_distance = simulate_train_loop_qrl(train1, train2, track, take_bypass_train1, take_bypass_train2)
        
        elif mode == "Random":
            current_distance = simulate_train_loop_random(train1, train2, track, take_bypass_train1)

        else:
            current_distance = simulate_train_loop_predictable(train1, train2, track, take_bypass_train1)
        
        if mode != "QRL":
            a_train2 = 0


        
        distances[i] = current_distance

        if current_distance > previous_distance:
            train1_reward = +1
            train2_reward = -1
        elif current_distance < previous_distance:
            train1_reward = -1
            train2_reward = +1
        else:
            train1_reward = 0
            train2_reward = 0

        # Update Q-values using Bellman equation
        train1_Q[a_train1] = train1_Q[a_train1] + alpha * (train1_reward + gamma * max(train1_Q) - train1_Q[a_train1])
        train2_Q[a_train2] = train2_Q[a_train2] + alpha * (train2_reward + gamma * max(train2_Q) - train2_Q[a_train2])
        
        # Calculate target probability based on Q-values
        p_train1 = train1_Q[0] / (train1_Q[0] + train1_Q[1]) if (train1_Q[0] + train1_Q[1]) != 0 else 0.5
        p_train2 = train2_Q[0] / (train2_Q[0] + train2_Q[1]) if (train2_Q[0] + train2_Q[1]) != 0 else 0.5
        
        # Train the quantum model using Q-values
        if p_train1 > 0:
            theta_train1 = update(theta_train1, p_train1)
        if p_train2 > 0:
            theta_train2 = update(theta_train2, p_train2)

        # Store probability of choosing loop for plotting
        M[i] = W(theta_train1)[0]
        A[i] = W(theta_train2)[0]

    # Save results for this parameter set
    results.append({
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'epochs': epochs,
        'final_distance': distances[-1],
        'mean_distance': np.mean(distances),
        'final_probability_train1': M[-1],
        'mean_probability_train1': np.mean(M),
        'final_probability_train2': A[-1],
        'mean_probability_train2': np.mean(A)
    })

# Display results in a readable format
import pandas as pd

df = pd.DataFrame(results)
print(df)

# Plot mean probability of taking loop for different epsilons
plt.figure(figsize=(10, 6))
for epsilon in epsilon_values:
    subset = df[df['epsilon'] == epsilon]
    plt.plot(subset['alpha'], subset['mean_probability_train1'], label=f'Train 1 - epsilon={epsilon}', marker='o')
    plt.plot(subset['alpha'], subset['mean_probability_train2'], label=f'Train 2 - epsilon={epsilon}', marker='x')

plt.title('Mean Probability of Taking Loop vs Alpha for Different Epsilons')
plt.xlabel('Alpha')
plt.ylabel('Mean Probability of Taking Loop')
plt.legend()
plt.grid(True)
plt.savefig('mean_probability_plot.png')

# ✅ Plot average distance over epochs
plt.figure(figsize=(10, 6))
# Calculate rolling average (window size = 10)
window = 10
avg_distances = np.convolve(distances, np.ones(window)/window, mode='valid')

# Plot the rolling average
plt.plot(range(window - 1, epochs), avg_distances, label='Average Distance', color='blue')
plt.title('Average Distance Between Trains Over Time')
plt.xlabel('Epoch')
plt.ylabel('Average Distance')
plt.legend()
plt.grid(True)
plt.savefig('average_distance_plot.png')

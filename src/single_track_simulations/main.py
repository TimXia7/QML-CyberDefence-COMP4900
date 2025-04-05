import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import pandas as pd
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

# Q-learning parameters
# Q-learning parameters
alpha_values = [0.1]
gamma_values = [0.99]

epsilon_values = [1.0] # Epsilon starts at 1.0, decays overtime
epsilon_end = 0.075   # Minimum exploration rate
decay_rate = 0.98   # How fast epsilon decreases

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
    track = SimpleTrack()
    train1 = Train(track, start_position=0)
    train2 = Train(track, start_position=7)

    previous_distance = calculate_distance(train1, train2, track)


    for i in range(epochs):
        print(f"Epoch: {i}")
        
        train1.set_position(0)
        train2.set_position(7)

        if np.random.rand() < epsilon:
            a_train1 = np.random.randint(2)
            a_train2 = np.random.randint(2)
        else:
            a_train1 = np.argmax(train1_Q)
            a_train2 = np.argmax(train2_Q)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * decay_rate)

        # Simulate train movement and get the reward
        if mode == "QRL":
            current_distance = simulate_train_loop_qrl(train1, train2, track, a_train1, a_train2)
        elif mode == "Random":
            current_distance = simulate_train_loop_random(train1, train2, track, a_train1)
        else:
            current_distance = simulate_train_loop_predictable(train1, train2, track, a_train1)
        

        distances[i] = current_distance

        if current_distance > previous_distance:
            train1_reward = current_distance-previous_distance
            train2_reward = previous_distance-current_distance
        elif current_distance < previous_distance:
            train1_reward = current_distance-previous_distance
            train2_reward = previous_distance-current_distance
        else:
            train1_reward = -1
            train2_reward = -1

        # Update Q-values using Bellman equation
        train1_Q[a_train1] = train1_Q[a_train1] + alpha * (train1_reward + gamma * max(train1_Q) - train1_Q[a_train1])
        
        # Calculate target probability based on Q-values
        p_train1 = train1_Q[0] / (train1_Q[0] + train1_Q[1]) if (train1_Q[0] + train1_Q[1]) != 0 else 0.5


        if p_train1 > 0:
            theta_train1 = update(theta_train1, p_train1)

        # Store probability of choosing loop for plotting
        M[i] = W(theta_train1)[0]

        if mode == "QRL":
            train2_Q[a_train2] = train2_Q[a_train2] + alpha * (train2_reward + gamma * max(train2_Q) - train2_Q[a_train2])

            total_q2 = sum(train2_Q)

            p_train2 = train2_Q[0] / (train2_Q[0] + train2_Q[1]) if (train2_Q[0] + train2_Q[1]) != 0 else 0.5

            if p_train2 > 0:
                theta_train2 = update(theta_train2, p_train2)

            # Store probability of choosing loop for plotting
            A[i] = W(theta_train2)[0]


    # Save results to graph
    results.append({
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'epochs': epochs,
        'final_distance': distances[-1],
        'mean_distance': np.mean(distances),
        'final_probability_train1_loop': M[-1],
        'mean_probability_train1_loop': np.mean(M),
        'final_probability_train2_loop': A[-1],
        'mean_probability_train2_loop': np.mean(A),
    })
# Plotting the probabilities for M and M-1
plt.plot(range(epochs), M, label="Train 1 - Loop Probability (M)", linestyle='-', marker='o', markersize=4, alpha=0.7)
plt.plot(range(epochs), 1 - M, label="Train 1 - Bypass Probability (1 - M)", linestyle='--', marker='x', markersize=4, alpha=0.7)

plt.title('Probability of Taking Loop (M) and Bypass (1 - M) Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Probability')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('loop_probability_plot_M_M1_best_model_control_test.png')
plt.show()


plt.figure(figsize=(10, 6))

plt.plot(range(epochs), distances, label='Distance', color='blue', alpha=0.6)

# Calculate and plot the average line
average_distance = np.mean(distances)
plt.axhline(y=average_distance, color='red', linestyle='--', label='Average Distance')

# Add title and labels
plt.title('Distance and Average Distance Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)


plt.savefig('distance_plot_best_model_control_test.png')
plt.show()

df = pd.DataFrame(results)
df.to_csv("q_learning_results_simple_best_model_control_test.csv", index=False)


plt.figure(figsize=(10, 6))

df.to_csv("q_learning_results.csv", index=False)
print("Results saved to q_learning_results.csv")
print(df)


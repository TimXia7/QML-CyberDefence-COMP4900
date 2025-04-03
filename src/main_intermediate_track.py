import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import pandas as pd
from train_simulation import *  

# 2-Qubit Variational Quantum Circuit 
dev = qml.device("default.qubit", wires=2, shots=1000)

@qml.qnode(dev)
def VQC(theta):
    
    qml.RX(theta[0], wires=0)
    qml.RY(theta[1], wires=0)
    
    qml.RX(theta[2], wires=1)
    qml.RY(theta[3], wires=1)

    # entanglement with CNOT
    qml.CNOT(wires=[0, 1])

    return qml.probs(wires=[0, 1])

# Only keep 3 states out of the max of 4 with 2 qubits
def map_VQC_results(theta):
    probs = VQC(theta)
    return np.array([probs[0], probs[1], probs[2] + probs[3]])

# Mean Squared Error Loss Function
def square_loss(X, Y):
    return np.mean((X - Y) ** 2) 

# Cost function
def cost(theta, p_target):
    return square_loss(map_VQC_results(theta), p_target)

# Optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.01)

# Training Function
def update(theta, p_target):
    for _ in range(100): 
        theta = opt.step(lambda v: cost(v, p_target), theta)
    return theta

# Q-learning parameters
alpha_values = [0.1]
gamma_values = [0.9]
epsilon_values = [0.3]
epoch_values = [100]
results = []
mode = "Random"

for alpha, gamma, epsilon, epochs in itertools.product(alpha_values, gamma_values, epsilon_values, epoch_values):
    print(f"Running with alpha={alpha}, gamma={gamma}, epsilon={epsilon}, epochs={epochs}")

    # Initialize Q-values and quantum model parameters
    theta_train1 = np.random.randn(10) 
    theta_train2 = np.random.randn(10)  
    train1_Q = [0, 0, 0]
    train2_Q = [0, 0, 0]
    distances = np.zeros(epochs)

    # Store probabilities for graphs
    M_loop = np.zeros(epochs)      
    M_bypass = np.zeros(epochs)    
    M_outerLoop = np.zeros(epochs) 

    A_loop = np.zeros(epochs)
    A_bypass = np.zeros(epochs)
    A_outerLoop = np.zeros(epochs)

    distances = np.zeros(epochs)
    # Initialize track and trains
    track = IntermediateTrack()
    train1 = Train(track, start_position=0)
    train2 = Train(track, start_position=7)

    previous_distance = calculate_distance(train1, train2, track)

    for i in range(epochs):
        # print(f"Epoch: {i}")

        # Epsilon-greedy action selection for Train 1
        # randomly pick choice, unless epsilon is higher, in which case pick what was learned
        if np.random.rand() < epsilon:
            a_train1 = np.random.randint(0, 3)  # Randomly pick 0, 1, or 2
        else:
            a_train1 = np.argmax(train1_Q)  # Choose best action from Q-table
            a_train2 = np.argmax(train2_Q)

        
        if mode == "QRL":
            current_distance = simulate_train_loop_qrl(train1, train2, track, a_train1, a_train2)
        elif mode == "Random":
            current_distance = simulate_train_loop_random(train1, train2, track, a_train1)
        else:
            current_distance = simulate_train_loop_predictable(train1, train2, track, a_train1)

        

        # Simulate train movement and get reward
        current_distance = simulate_train_loop_random(train1, train2, track, a_train1)
        distances[i] = current_distance

        # Determine Reward
        if current_distance > previous_distance:
            train1_reward = +1
        elif current_distance < previous_distance:
            train1_reward = -1
        else:
            train1_reward = 0
            train2_reward = 0

        # Update Q-values using Bellman's equation
        train1_Q[a_train1] = train1_Q[a_train1] + alpha * (train1_reward + gamma * max(train1_Q) - train1_Q[a_train1])

        # Calculate probabilities for each action based on Q-values
        total_q = sum(train1_Q)

        # Choose the corresponding q-value, unless they are all 0, then pick randomly
        if total_q != 0:
            p_train1_loop = train1_Q[0] / total_q
            p_train1_bypass = train1_Q[1] / total_q
            p_train1_outerLoop = train1_Q[2] / total_q
        else:
            p_train1_loop = 1 / 3
            p_train1_bypass = 1 / 3
            p_train1_outerLoop = 1 / 3

        # DEBUG: Print target probabilities for Train 1
        # print(f"Target probabilities for Train 1,  Loop: {p_train1_loop}, Bypass: {p_train1_bypass}, Outer Loop: {p_train1_outerLoop}")

        # Update the quantum model based on the target probabilities
        theta_train1 = update(theta_train1, [p_train1_loop, p_train1_bypass, p_train1_outerLoop])

        # Store probability of choosing each action for plotting
        action_probs = map_VQC_results(theta_train1)
        M_loop[i] = action_probs[0]  # Probability of taking the normal loop
        M_bypass[i] = action_probs[1]  # Probability of taking the bypass
        M_outerLoop[i] = action_probs[2]  # Probability of taking the outer loop



    # Save results to graph
    results.append({
        'alpha': alpha,
        'gamma': gamma,
        'epsilon': epsilon,
        'epochs': epochs,
        'final_distance': distances[-1],
        'mean_distance': np.mean(distances),
        'final_probability_loop': M_loop[-1],
        'mean_probability_loop': np.mean(M_loop),
        'final_probability_bypass': M_bypass[-1],
        'mean_probability_bypass': np.mean(M_bypass),
        'final_probability_outerLoop': M_outerLoop[-1],
        'mean_probability_outerLoop': np.mean(M_outerLoop),
    })

# Graph results
df = pd.DataFrame(results)
print(df)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), M_bypass, label="Bypass", color='green', linestyle='--')
plt.plot(range(epochs), M_loop, label="Loop", color='blue', linestyle='-')
plt.plot(range(epochs), M_outerLoop, label="Outer Loop", color='red', linestyle='-.')

plt.title('Train 1 Action Probabilities Over Time')
plt.xlabel("Epoch")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.savefig('train_action_probabilities.png')


plt.figure(figsize=(10, 6))
plt.plot(range(epochs), distances, label='Distance', color='blue', alpha=0.6)
window = 10
avg_distances = np.convolve(distances, np.ones(window)/window, mode='valid')
plt.plot(range(window - 1, epochs), avg_distances, label='Average Distance', color='red', linestyle='--')
plt.title('Distance Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)
plt.savefig('distance_plot.png')
plt.show()


plt.show()

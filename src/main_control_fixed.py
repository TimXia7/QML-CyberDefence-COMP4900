import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from train_simulation import *

def simulate_train_loop_control_fixed(train1, train2, track):
    start_position = train1.current_node.position

    while True:
        train1.move(random.randint(0, 1))
        train2.move(0)

        if train1.current_node.position == start_position:
            break

    return calculate_distance(train1, train2, track)

# Simulation parameters
epochs = 1000
track = SimpleTrack()
train1 = Train(track, start_position=0)
train2 = Train(track, start_position=7)

distances = np.zeros(epochs)

# Run the simulation for 300 epochs
for i in range(epochs):
    distances[i] = simulate_train_loop_control_fixed(train1, train2, track)
    # train1.set_position(0)
    # train2.set_position(7)

# Save results to graph
results = []
results.append({
    'epochs': epochs,
    'final_distance': distances[-1],
    'mean_distance': np.mean(distances),
})

# Plot distance over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), distances, label='Distance', color='blue', alpha=0.6)

# Calculate and plot the average distance line
average_distance = np.mean(distances)
plt.axhline(y=average_distance, color='red', linestyle='--', label='Average Distance')

plt.title('Distance and Average Distance Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)
plt.savefig('distance_plot_control_test.png')
plt.show()

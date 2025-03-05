import pennylane as qml
import numpy as np
from variational_circuit import variational_circuit

n, m = 3, 2

take_loop, take_bypass = 0, 1

T = np.zeros((n, n, m), dtype=np.float32)
T[0][0][take_loop], T[0][0][take_bypass] = 0.5, 0.5
T[0][1][take_loop], T[0][1][take_bypass] = 0.5, 0.5

R = np.zeros((n, n, m), dtype=np.int8)
R[0][0][take_loop], R[0][0][take_bypass] = 0, 4
R[0][1][take_loop], R[0][1][take_bypass] = 0, 2

TerminalStates = {1, 2}

states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

theta_x, theta_y = np.pi / 4, 0.78539816339
print(variational_circuit(theta_x, theta_y))

Q = np.zeros((n, m))

alpha, gamma, epsilon = 0.1, 0.9, 0.1

for episode in range(100):
    state = 0
    done = False
    while not done:
        action = np.random.randint(m) if np.random.rand() < epsilon else np.argmax(Q[state])
        next_state = np.random.choice(n, p=T[state, :, action])
        reward = R[state, next_state, action]
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        if state in TerminalStates:
            done = True

print("Trained Q-table:", Q)

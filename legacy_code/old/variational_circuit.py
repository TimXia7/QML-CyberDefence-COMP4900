
# This file holds the Variational circuit and related computations

import pennylane as qml
import numpy as np

# Create a "blank" quantum circuit with an input of 1 qubit
# By default, according to PennyLane documentation, the input value is ket 0
dev = qml.device("default.qubit", wires=1)

# theta should be 0 < theta < 2pi
# These should vary based on the ML portion
# Note, they can be both decimals or a more percise value via math libraries
# theta_x is pi/2, theta_y is pi/4 (example values)
theta_x = np.pi / 2 
theta_y = np.pi / 2

# Define quantum device "dev." To my knowledge, this is just PennyLane syntax. 
# This function that will run the circuit from before.
# Here, we add an X and Y Gate for our 2 parameters in the variational circuit
@qml.qnode(dev)
def variational_circuit(theta_x, theta_y):
    qml.RX(theta_x, wires=0) 
    qml.RY(theta_y, wires=0)
    return qml.expval(qml.PauliZ(0))  # Measure expectation value

# "circuit(theta_x, theta_y)" contains the result of the circuit, the chance to take the pass or not
# e.g. 1.0 means gurantee to take pass, 0.0 gurantee to take loop, 0.5 means even chance
# uncomment the line below if you want to test within this file
print(variational_circuit(theta_x, theta_y))

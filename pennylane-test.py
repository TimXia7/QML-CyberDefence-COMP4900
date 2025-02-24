import pennylane as qml
import numpy as np

# Create a quantum device with 1 qubit
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)  # Apply Hadamard gate
    return qml.probs(wires=0)  # Return probabilities

# Execute the circuit
output = circuit()
print("Quantum Circuit Output:", output)

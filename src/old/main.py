
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev1 = qml.device("default.qubit", wires=1, shots=1000)


@qml.qnode(dev1)
def W(theta):
    qml.RX(theta[0], wires=0)
    qml.RY(theta[1], wires=0)
    # return probabilities of computational basis states
    return qml.probs(wires=[0])

def square_loss(X, Y):
    loss = 0
    for x, y in zip(X, Y):
        loss = loss + (x - y) ** 2
    return loss / len(X)

def cost(theta, p):
    #  p is prob. of ket zero, 1-p is prob of ket 1
    return square_loss(W(theta), [p, 1-p])

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.01)
# set the number of steps
steps = 100
# set the initial theta value of variational circuit state
theta = np.random.randn(2)
# print("Initial probs. of basis states: {}".format(W(theta)))

def update(theta, p):
    # p = probability
    for i in range(steps):
        # update the circuit parameters
        theta = opt.step(lambda v: cost(v, p), theta)
        
    # print("Probs. of basis states: {}".format(W(theta)))
    return theta


# init theta to random values
theta = np.random.randn(2)
# number of iterations
epochs = 100
# measurement results (probs. of ket zero and ket one)
M = np.zeros(epochs)
# environment probability
p = 0.5
# rewards
R = [4, 2]
# Q-value (total reward associated with actions)
Q = [0, 0]
# learning rate
alpha = 0.5
# main loop
for i in range(epochs):
    # random action
    a = np.random.randint(2, size=1)[0]
    # determine reward
    num = np.random.random(1)[0]
    r = 0
    if num > p:
        r = R[a]
    # Bellman equation
    Q[a] = (1-alpha)*Q[a] + alpha*(r + Q[a])
    # print("action: ", a, " reward: ", r, " Q: ", Q)
    if (Q[0]+Q[1]) > 0:
        a=0
        theta = update(theta, Q[0]/(Q[0]+Q[1]))
    M[i] = W(theta)[0]



plt.plot(M, 'bs', label='take loop')
plt.plot(1-M, 'g^', label='take bypass')
plt.xlabel('Epoch')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.show()
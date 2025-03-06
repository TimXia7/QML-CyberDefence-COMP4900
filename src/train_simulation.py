
import random

# All classes for the simulation are in this one file for simplicity. This is because, a. it's not very complex
# and b. it'll act as a black box for the other, more complex portions of the project

# i.e. gives outputs like train positions after 1 loop, or current distance between trains, elc.
# And the system that learns should try to take the most optimal decision, to take loop or not. 

# The track is represented as a linked list.
# Node represents each numbered section of the track
class Node:
    def __init__(self, position):
        self.position = position
        self.next = None  
        self.bypass = None  

class Track:
    def __init__(self):
        self.nodes = {i: Node(i) for i in range(9)}

        # Regular train track that includes the loop (0-7)
        self.nodes[0].next = self.nodes[1]
        self.nodes[1].next = self.nodes[2]
        self.nodes[2].next = self.nodes[3]
        self.nodes[3].next = self.nodes[4]
        self.nodes[4].next = self.nodes[5]
        self.nodes[5].next = self.nodes[6]
        self.nodes[6].next = self.nodes[7]
        self.nodes[7].next = self.nodes[0]

        # Initiate Bypass with optional bypass variable (node 8)
        self.nodes[2].bypass = self.nodes[8]
        self.nodes[8].next = self.nodes[6]


# Train class for agents and adversaries 
class Train:
    def __init__(self, track, start_position):
        self.track = track
        self.current_node = track.nodes[start_position]

    # Move the train to the next node (node.next)
    # If the boolean parameter take_bypass = True, then take the bypass if avaliable
    def move(self, take_bypass):
        if self.current_node.position == 2 and take_bypass:
            self.current_node = self.current_node.bypass 
        else:
            self.current_node = self.current_node.next

# Calculate the distance between the 2 trains in a similar way to the paper
# note: Since train 2 always takes the loop, distance is calculated this way
def calculate_distance(train1, train2, track):
    distance = 0
    current = train2.current_node
    while current != train1.current_node:
        current = current.next  
        distance += 1
    return distance

# Start the movement of the trains until train1 gets back to node 0
# IMPORTANT: parameter, train1_decision_func, determines if train1 takes the loop or not, 
# function returns the distance between the 2 trains after the loop
def simulate_train_loop(train1, train2, track, take_bypass):
    start_position = train1.current_node.position
    while True:

        if (train1.current_node.position == 2 ):
            if (take_bypass):
                print("The agent decided to take the bypass!")
            else:
                print("The agent decided to take the loop!")

        train1.move(take_bypass)
        train2.move(False)  

        if train1.current_node.position == start_position:
            break

    return calculate_distance(train1, train2, track)


# Randomly returns true or false
# This is because right now, train1 chooses to take the bypass randomly
def rand_bool():
    return random.random() < 0.5  






# Start of main method for testing, can be removed later:

track = Track()
train1 = Train(track, start_position=0)
train2 = Train(track, start_position=7)

# 10 loops for testing
previous_distance = None 
for loop in range(1, 11): 
    distance = simulate_train_loop(train1, train2, track, rand_bool())
    
    # Calculate difference from previous loop
    if (loop != 1):
        difference = distance - previous_distance
    else:
        difference = "N/A"
    
    # Print results
    print(f"LOOP {loop}: Final distance = {distance}, Difference from last loop = {difference}")
    print(f"Agent's position:     {train1.current_node.position}")
    print(f"Adversary's position: {train2.current_node.position} \n")
    
    # Update previous distance for next iteration
    previous_distance = distance

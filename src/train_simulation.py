
import random
from collections import deque

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

    @property
    def neighbors(self):
        neighbors = []
        if self.next:
            neighbors.append(self.next)
        if self.bypass:
            neighbors.append(self.bypass)
        return neighbors

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


def calculate_distance(train1, train2, track):
    # BFS for shortest path calculation
    queue = deque([(train2.current_node, 0)])
    visited = set()
    
    while queue:
        node, distance = queue.popleft()
        
        if node == train1.current_node:
            return distance
        
        if node not in visited:
            visited.add(node)
            for neighbor in node.neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))

    return float('inf')


# Start the movement of the trains until train1 gets back to node 0
# IMPORTANT: parameter, take_bypass, determines if train1 takes the loop or not, 
# function returns the distance between the 2 trains after the loop
def simulate_train_loop(train1, train2, track, take_bypass):
    start_position = train1.current_node.position
    while True:

        # if (train1.current_node.position == 2 ):
        #     if (take_bypass):
        #         print("The agent decided to take the bypass!")
        #     else:
        #         print("The agent decided to take the loop!")

        train1.move(take_bypass)
        train2.move(False)  

        if train1.current_node.position == start_position:
            break

    return calculate_distance(train1, train2, track)

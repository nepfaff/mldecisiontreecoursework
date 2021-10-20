import matplotlib.pyplot as plt

# Example basic tree 

class Node:
    def __init__(self, is_leaf, attribute, value, left, right, label):
        self.is_leaf = is_leaf
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.label = label

Node7 = Node(True, 'B', 0, None, None, 3)
Node6 = Node(False, 'C', 21, None, None, None)
Node5 = Node(False, 'B', 21, None, None, None)
Node4 = Node(False, 'B', 21, None, Node7, None)
Node3 = Node(False, 'D', 21, Node5, Node6, None)
Node2 = Node(False, 'B', 21, None, Node4, None)
Node1 = Node(False, 'A', 54, Node2, Node3, None)



# For creating a line between nodes 

def joinNodes(x1, y1, x2, y2, ax):
    ax.plot([x1, x2], [y1, y2])

# Function that plots Nodes recursivly  

def plotNode(x, y, Node, ax, distance):
# Defining variables for creating the range of the figure 
    xL_left, xL_right, xR_left, xR_right, y_left, y_right = 0, 0, 0, 0, 0, 0

# Creating text visible for each node 
    boxText = '[' + Node.attribute + '<' + str(Node.value) + ']'

# Checking to see if leaf node 
    if Node.is_leaf == True:
        boxText = 'leaf: ' + str(Node.label)

# Plotting node with a box around it 
    ax.text(x, y, boxText, color='red', 
        bbox=dict(facecolor='white', edgecolor='red'))

# Checking for left and right nodes and recusivly calling function
    if Node.left != None:
       xL_left, xL_right, y_left = plotNode((x-distance), (y-5), Node.left, ax, (distance/2))
       joinNodes(x, y, (x-distance), (y-5), ax)
    if Node.right != None:
        xR_left, xR_right, y_right = plotNode((x+distance), (y-5), Node.right, ax, (distance/2))
        joinNodes(x, y, (x+distance), (y-5), ax)

# Building boarders for figure
    if xL_left == 0:
        xL = x
    else:
        xL = xL_left

    if xR_right == 0:
        xR = x
    else: 
        xR = xR_right

    return xL, xR, min(y_left, y_right)

# Function that takes the top node of the tree and plots the whole tree 
def plotTree(Tree):
# Defining figure information
    start_location = [0, 0]
    fig, ax = plt.subplots(figsize=(12, 6))

# Calling recursive node plotting software
    x_left, x_right, y_depth = plotNode(start_location[0], start_location[1], Tree, ax, 50)

# Assinging figure boarders and information

    if x_left == 0:
        x_left = -50
    if x_right ==0:
        x_right = 50

    if y_depth == 0:
        y_depth = -50

    plt.xlim([x_left, x_right])
    plt.ylim([5, y_depth])
    plt.gca().invert_yaxis()
    ax.axis('off')
    plt.show()

#   Saving figure to file  
    plt.savefig('visualisation/tree.png')


# calling functions for testing
plotTree(Node1)
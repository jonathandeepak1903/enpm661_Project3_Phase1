import cv2
import numpy as np
import math
import heapq
import time


def generate_map(clearance=5):
    print("generating map")
    canvas = np.ones((550, 1250, 3), dtype="uint8") * 255  # White background
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)

    # Draw wall
    cv2.rectangle(canvas, (0, 0), (1200, 500), BLACK, 1)

    # Draw obstacles
    cv2.rectangle(canvas, (100 - clearance, 0), (175 + clearance, 400 + clearance), RED, -1)  # Upper Rectangle
    cv2.rectangle(canvas, (275 - clearance, 500), (350 + clearance, 100 + clearance), RED, -1)  # Lower Rectangle
    # Hexagon
    vertices = np.array([[650 - clearance, 100 - clearance], [800 + clearance, 175 - clearance],
                         [800 + clearance, 325 + clearance], [650 + clearance, 400 + clearance],
                         [500 - clearance, 325 + clearance], [500 - clearance, 175 - clearance]], dtype=np.int32)
    cv2.fillPoly(canvas, [vertices], RED)
    # Letter C
    cv2.rectangle(canvas, (1020 - clearance, 50 - clearance), (1100 + clearance, 450 + clearance), RED, -1)
    cv2.rectangle(canvas, (900 - clearance, 375 - clearance), (1020 + clearance, 450 + clearance), RED, -1)
    cv2.rectangle(canvas, (900 - clearance, 50 - clearance), (1020 + clearance, 125 + clearance), RED, -1)
    print("map generated")
    return canvas

# Define obstacles using half-plane equations
obstacles = [
    # Upper Rectangle
    lambda x, y: (x - 100) >= -clearance and (x - 175) <= clearance and (y - 0) >= -clearance and (y - 400) <= clearance,
    # Lower Rectangle
    lambda x, y: (x - 275) >= -clearance and (x - 350) <= clearance and (y - 500) >= clearance and (y - 100) <= clearance,
    # Hexagon
    lambda x, y: (x - 650) >= -clearance and (x - 800) <= clearance and (y - 100) >= -clearance and (y - 400) <= clearance and (x - 500) >= -clearance and (x - 800) <= clearance and (y - 175) >= -clearance and (y - 325) <= clearance,
    # Letter C
    lambda x, y: (x - 1020) >= -clearance and (x - 1100) <= clearance and (y - 50) >= -clearance and (y - 450) <= clearance,
    lambda x, y: (x - 900) >= -clearance and (x - 1020) <= clearance and (y - 375) >= -clearance and (y - 450) <= clearance,
    lambda x, y: (x - 900) >= -clearance and (x - 1020) <= clearance and (y - 50) >= -clearance and (y - 125) <= clearance,
    # Add wall equations
    lambda x, y: y <= 0 or y >= 500 or x <= 0 or x >= 1200,
]

# Define robot radius and clearance
robot_radius = 5
clearance = 5

# Define action set functions
def go_straight(x, y, theta, step_size):
    new_x = x + step_size * math.cos(math.radians(theta))
    new_y = y + step_size * math.sin(math.radians(theta))
    new_theta = theta
    return new_x, new_y, new_theta

def turn_left(x, y, theta, step_size):
    new_x = x
    new_y = y
    new_theta = (theta - 30) % 360
    return new_x, new_y, new_theta

def turn_right(x, y, theta, step_size):
    new_x = x
    new_y = y
    new_theta = (theta + 30) % 360
    return new_x, new_y, new_theta

def go_straight_and_turn_left(x, y, theta, step_size):
    new_x, new_y, new_theta = go_straight(x, y, theta, step_size)
    new_x, new_y, new_theta = turn_left(new_x, new_y, new_theta, step_size)
    return new_x, new_y, new_theta

def go_straight_and_turn_right(x, y, theta, step_size):
    new_x, new_y, new_theta = go_straight(x, y, theta, step_size)
    new_x, new_y, new_theta = turn_right(new_x, new_y, new_theta, step_size)
    return new_x, new_y, new_theta

# Define heuristic function (Euclidean distance)
def heuristic(x, y, goal_x, goal_y):
    return math.sqrt((x - goal_x)**2 + (y - goal_y)**2)

# Define function to check if a node is in the obstacle space
def is_in_obstacle(x, y):
    for obstacle in obstacles:
        if obstacle(x, y):
            return True
    return False

# Define function to check if a node is in the visited region
def is_visited(x, y, theta, visited):
    i = int(x // 0.5)
    j = int(y // 0.5)
    k = int(theta // 30)
    return visited[i][j][k]

# Define function to mark a node as visited
def mark_visited(x, y, theta, visited):
    i = int(x // 0.5)
    j = int(y // 0.5)
    k = int(theta // 30)
    visited[i][j][k] = True

# Define A* algorithm
def a_star(start_x, start_y, start_theta, goal_x, goal_y, goal_theta, step_size):
    print("A* started to find path")
    # Initialize Open list and Closed list
    open_list = []
    closed_list = set()

    # Initialize start node
    start_node = (start_x, start_y, start_theta, 0, None)
    heapq.heappush(open_list, (heuristic(start_x, start_y, goal_x, goal_y), start_node))

    # Initialize visited matrix
    visited = [[[False for k in range(12)] for j in range(2500)] for i in range(1100)]

    # A* algorithm loop
    while open_list:
        # Pop the node with the lowest cost from the Open list
        _, current_node = heapq.heappop(open_list)
        current_x, current_y, current_theta, cost_to_come, parent = current_node

        # Check if the current node is the goal node
        if math.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2) <= 1.5:
            # Backtrack to get the optimal path
            optimal_path = []
            node = current_node
            while node:
                optimal_path.append(node[:3])
                node = node[4]
            optimal_path.reverse()
            return optimal_path

        # Mark the current node as visited
        closed_list.add(current_node)
        mark_visited(current_x, current_y, current_theta, visited)

        # Generate successor nodes
        for action in [go_straight, turn_left, turn_right, go_straight_and_turn_left, go_straight_and_turn_right]:
            new_x, new_y, new_theta = action(current_x, current_y, current_theta, step_size)

            # Check if the new node is in the obstacle space or has been visited before
            if is_in_obstacle(new_x, new_y) or is_visited(new_x, new_y, new_theta, visited):
                continue

            # Calculate the cost of the new node
            new_cost_to_come = cost_to_come + step_size
            new_node = (new_x, new_y, new_theta, new_cost_to_come, current_node)

            # Check if the new node is in the Open list and has a lower cost
            in_open_list = False
            for idx, open_node in enumerate(open_list):
                if open_node[1][:3] == new_node[:3]:
                    if open_node[1][3] > new_cost_to_come:
                        open_list[idx] = (new_cost_to_come + heuristic(new_x, new_y, goal_x, goal_y), new_node)
                    in_open_list = True
                    break

            # Add the new node to the Open list if it is not in the list or has a lower cost
            if not in_open_list:
                heapq.heappush(open_list, (new_cost_to_come + heuristic(new_x, new_y, goal_x, goal_y), new_node))

    # No path found
    return []

def visualize_path(canvas, optimal_path):
    print("Video generation started")
    start_time = time.time()
    # Draw optimal path on the canvas
    for node in optimal_path:
        x, y, theta = node
        cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Save animation as a video
    video = cv2.VideoWriter('animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1250, 550))
    for node in optimal_path:
        x, y, theta = node
        cv2.circle(canvas, (int(x), int(y)), 3, (0, 255, 0), -1)
        video.write(canvas)
    video.release()
    end_time = time.time()
    print("Video generated in {} seconds".format(end_time - start_time))
    # Display the final canvas
    cv2.imshow("Optimal Path", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Get user inputs
start_x = float(input("Enter start x coordinate: "))
start_y = float(input("Enter start y coordinate: "))
start_theta = int(input("Enter start orientation (in degrees, multiple of 30): "))

goal_x = float(input("Enter goal x coordinate: "))
goal_y = float(input("Enter goal y coordinate: "))
goal_theta = int(input("Enter goal orientation (in degrees, multiple of 30): "))

step_size = float(input("Enter step size (between 1 and 10): "))

# Generate the map
canvas = generate_map(clearance)

# Run A* algorithm
start_time = time.time()
optimal_path = a_star(start_x, start_y, start_theta, goal_x, goal_y, goal_theta, step_size)
end_time = time.time()
print("Optimal path found using a star in {} seconds".format(end_time - start_time))
visualize(optimal_path,canvas)

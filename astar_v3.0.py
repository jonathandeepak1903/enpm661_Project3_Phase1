import cv2
import numpy as np
import heapq
import time
import cProfile
# Define colors
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

# Define actions
ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
COSTS = [1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4]

class Node:
    def __init__(self, position, cost_to_come, cost_to_go):
        self.position = position
        self.cost_to_come = cost_to_come
        self.cost_to_go = cost_to_go

    def total_cost(self):
        return self.cost_to_come + self.cost_to_go

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.position == other.position
        return NotImplemented

def astar(start, goal, canvas):
    open_list = [Node(start, 0, heuristic(start, goal))]  # Initialize open_list with nodes
    closed_list = []

    while open_list and goal not in closed_list:
        current_node = min(open_list)  # Use the comparison methods defined in the Node class
        open_list.remove(current_node)
        closed_list.append(current_node.position)

        if current_node.position == goal:  # Check if the current node is the goal
            return current_node.position  # SUCCESS: backtrack can be done by following parent nodes

        for action, cost in zip(ACTIONS, COSTS):
            next_x = current_node.position[0] + action[0]
            next_y = current_node.position[1] + action[1]
            next_node = Node((next_x, next_y), current_node.cost_to_come + cost, heuristic((next_x, next_y), goal))
            if next_node not in closed_list and is_valid_point(next_node):
                open_list.append(next_node)

    return None  # FAILURE: goal node was not reached

def is_valid_point(node):
    x, y = node.position
    return 0 <= x < 1200 and 0 <= y < 500 and not is_obstacle(node.position)

def generate_map(clearance=5):
    print("Generating map")
    canvas = np.ones((500, 1200, 3), dtype="uint8") * 255  # White background

    # Draw wall
    cv2.rectangle(canvas, (0, 0), (1200, 500), BLACK, 1)

    # Draw obstacles with clearance
    # Upper Rectangle
    cv2.rectangle(canvas, (100 - clearance, 0 - clearance), (175 + clearance, 400 + clearance), RED, -1)
    # Lower Rectangle
    cv2.rectangle(canvas, (275 - clearance, 500), (350 + clearance, 100 + clearance), RED, -1)
    # Hexagon
    vertices = np.array([[650 - clearance, 100 - clearance], [800 + clearance, 175 - clearance],
                         [800 + clearance, 325 + clearance], [650 + clearance, 400 + clearance],
                         [500 - clearance, 325 + clearance], [500 - clearance, 175 - clearance]], dtype=np.int32)
    cv2.fillPoly(canvas, [vertices], RED)
    # Letter C
    cv2.rectangle(canvas, (1020 - clearance, 50 - clearance), (1100 + clearance, 450 + clearance), RED, -1)
    cv2.rectangle(canvas, (900 - clearance, 375 - clearance), (1020 + clearance, 450 + clearance), RED, -1)
    cv2.rectangle(canvas, (900 - clearance, 50 - clearance), (1020 + clearance, 125 + clearance), RED, -1)

    # Draw black borders to define clearance
    # Upper Rectangle
    cv2.rectangle(canvas, (100 - clearance, 0 - clearance), (175 + clearance, 400 + clearance), BLACK, 1)
    # Lower Rectangle
    cv2.rectangle(canvas, (275 - clearance, 500), (350 + clearance, 100 + clearance), BLACK, 1)
    # Hexagon
    cv2.polylines(canvas, [vertices], isClosed=True, color=BLACK, thickness=1)
    # Letter C
    # Side rectangle
    cv2.rectangle(canvas, (1020 - clearance, 50 - clearance), (1100 + clearance, 450 + clearance), BLACK, 1)
    # Lower Rectangle
    cv2.rectangle(canvas, (900 - clearance, 375 - clearance), (1020 - clearance, 450 + clearance), BLACK, 1)
    # Upper Rectangle
    cv2.rectangle(canvas, (900 - clearance, 50 - clearance), (1020 - clearance, 125 + clearance), BLACK, 1)

    print("Map generated")
    return canvas


def is_obstacle(point, clearance=5):
    x, y = point
    robot_radius = 5  # Assuming a robot radius of 5 units

    # Define obstacles using half-plane equations
    obstacles = [
        # Upper Rectangle
        lambda x, y: 100 - clearance <= x <= 175 + clearance and 0 - clearance <= y <= 400 + clearance,
        # Lower Rectangle
        lambda x, y: 275 - clearance <= x <= 350 + clearance and 500 - clearance <= y <= 100 + clearance,
        # Hexagon
        lambda x, y: cv2.pointPolygonTest(
            np.array([[650, 100], [800, 175], [800, 325], [650, 400], [500, 325], [500, 175]], dtype=np.int32), (x, y),
            True) >= -clearance,
        # Letter C
        lambda x, y: 1020 - clearance <= x <= 1100 + clearance and 50 - clearance <= y <= 450 + clearance,
        lambda x, y: 900 - clearance <= x <= 1020 + clearance and 375 - clearance <= y <= 450 + clearance,
        lambda x, y: 900 - clearance <= x <= 1020 + clearance and 50 - clearance <= y <= 125 + clearance,
        # Wall equations
        lambda x, y: y <= clearance or y >= 500 - clearance or x <= clearance or x >= 1200 - clearance,
    ]

    # Check if the point is inside any obstacle region
    for obstacle in obstacles:
        if obstacle(x, y):
            return True

    # Check if the point is within robot_radius distance from any obstacle
    for obstacle in obstacles:
        for x_offset in range(-robot_radius, robot_radius + 1):
            for y_offset in range(-robot_radius, robot_radius + 1):
                if obstacle(x + x_offset, y + y_offset):
                    return True

    return False


def heuristic(node, goal):
    return np.sqrt((goal[0] - node[0]) ** 2 + (goal[1] - node[1]) ** 2)




def visualize_path(canvas, path):
    print("Video generation started")
    start_time = time.time()
    height, width, _ = canvas.shape
    video = cv2.VideoWriter("optimal_path.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))

    for node in path:
        if node == path[0]:
            cv2.circle(canvas, node, 5, YELLOW, -1)
        else:
            prev_node = path[path.index(node) - 1]
            cv2.line(canvas, prev_node, node, BLACK, 10)
            cv2.circle(canvas, node, 5, YELLOW, -1)

        video.write(canvas)

        if node != path[0]:
            cv2.circle(canvas, prev_node, 5, BLACK, -1)

    end_time = time.time()
    print("Time taken to generate video: " + str(end_time - start_time) + " seconds")
    video.release()
    print("Video generation completed.")

def main():
    start_x, start_y = map(int, input("Enter start x and y coordinates separated by a comma (x, y): ").split(','))
    start = Node((start_x, start_y), 0, 0)

    goal_x, goal_y = map(int, input("Enter goal x and y coordinates separated by a comma (x, y): ").split(','))
    goal = Node((goal_x, goal_y), 0, 0)

    canvas = generate_map()

    if not (0 <= start.position[0] < 1200 and 0 <= start.position[1] < 500) or not (0 <= goal.position[0] < 1200 and 0 <= goal.position[1] < 500):
        print("Invalid start or goal coordinates. Please try again.")
        return

    start_time = time.time()
    path = astar(start.position, goal.position, canvas)
    end_time = time.time()

    if not path:
        print("No path found between start and goal.")
    else:
        print(f"Path found in {end_time - start_time:.2f} seconds.")
        visualize_path(canvas.copy(), path)


if __name__ == "__main__":
    cProfile.run("main()", sort="cumulative")

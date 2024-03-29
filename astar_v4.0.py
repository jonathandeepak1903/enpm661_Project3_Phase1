import cv2
import numpy as np
import heapq
import time
import cProfile
import profile
import subprocess
from line_profiler import LineProfiler
import pstats
import matplotlib.path
from matplotlib.path import Path as mplPath

BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

# Define actions
ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
COSTS = [1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4]



def generate_map():
    print("Generating map...")
    canvas = np.ones((550, 1250, 3), dtype="uint8") * 255  # White background

    # Draw wall
    cv2.rectangle(canvas, (0, 0), (1200, 500), (0, 0, 0), 1)

    # Draw obstacles
    cv2.rectangle(canvas, (100, 0), (175, 400), (0, 0, 255), -1)  # Upper Rectangle
    cv2.rectangle(canvas, (275, 500), (350, 100), (0, 0, 255), -1)  # Lower Rectangle
    # Hexagon
    vertices = np.array([[650, 100], [800, 175], [800, 325], [650, 400], [500, 325], [500, 175]], dtype=np.int32)
    cv2.fillPoly(canvas, [vertices], (0, 0, 255))
    # Letter C
    cv2.rectangle(canvas, (1020, 50), (1100, 450), (0, 0, 255), -1)
    cv2.rectangle(canvas, (900, 375), (1020, 450), (0, 0, 255), -1)
    cv2.rectangle(canvas, (900, 50), (1020, 125), (0, 0, 255), -1)

    print("Map generated")
    # cv2.imshow("Map", canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return canvas



def is_obstacle(x, y, clearance):
    # Define obstacle boundaries including clearance for rectangular shapes
    obstacles = [
        [(100 - clearance, 0 - clearance), (175 + clearance, 400 + clearance)],  # Upper Rectangle
        [(275 - clearance, 100 - clearance), (350 + clearance, 500 + clearance)],  # Lower Rectangle
        [(1020 - clearance, 50 - clearance), (1100 + clearance, 450 + clearance)],  # C shape part 1
        [(900 - clearance, 375 - clearance), (1020 + clearance, 450 + clearance)],  # C shape part 2
        [(900 - clearance, 50 - clearance), (1020 + clearance, 125 + clearance)]   # C shape part 3
    ]

    # Check if point is within any rectangular obstacle
    for top_left, bottom_right in obstacles:
        if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
            return True

    # Hexagon check requires a different approach
    hexagon_vertices = np.array([[650, 100], [800, 175], [800, 325], [650, 400], [500, 325], [500, 175]], dtype=np.int32)
    hex_path = mplPath(hexagon_vertices)
    if hex_path.contains_point((x, y), radius=clearance):
        return True

    # Wall boundary checks
    if x <= 0 + clearance or x >= 1200 - clearance or y <= 0 + clearance or y >= 500 - clearance:
        return True

    return False


def heuristic(node, goal):
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    straight_cost = 1.0
    diagonal_cost = 1.4
    return (straight_cost * (dx + dy) + (diagonal_cost - 2 * straight_cost) * min(dx, dy))


def astar(start, goal, canvas, clearance, step_size):
    print("A* Algorithm started to find path")

    # Define actions based on step size
    ACTIONS = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
               (step_size, step_size), (-step_size, step_size),
               (step_size, -step_size), (-step_size, -step_size)]
    COSTS = [step_size] * 4 + [step_size * np.sqrt(2)] * 4  # Cost for straight and diagonal moves

    open_list = []
    closed_set = set()
    parent_node_dict = {}
    cost_to_come = {}
    explored_nodes = []

    heapq.heappush(open_list, (0, start))
    cost_to_come[start[:2]] = 0
    parent_node_dict[start[:2]] = None

    while open_list:
        _, current = heapq.heappop(open_list)
        current_x, current_y, _ = current

        if (current_x, current_y) == goal[:2]:
            path = []
            while current != start[:2]:
                path.append(current)
                current = parent_node_dict[current[:2]]
            return path[::-1], explored_nodes, parent_node_dict

        closed_set.add((current_x, current_y))
        explored_nodes.append((current_x, current_y))

        for dx, dy in ACTIONS:
            next_x, next_y = current_x + dx, current_y + dy
            if 0 <= next_x < canvas.shape[1] and 0 <= next_y < canvas.shape[0] and not is_obstacle(next_x, next_y,
                                                                                                   clearance):
                next_cost = cost_to_come[(current_x, current_y)] + COSTS[ACTIONS.index((dx, dy))]
                if (next_x, next_y) not in cost_to_come or next_cost < cost_to_come[(next_x, next_y)]:
                    cost_to_come[(next_x, next_y)] = next_cost
                    parent_node_dict[(next_x, next_y)] = (current_x, current_y)
                    heapq.heappush(open_list, (next_cost + heuristic((next_x, next_y), goal[:2]),
                                               (next_x, next_y, 0)))  # Assuming 0 as a placeholder for theta

    return [], explored_nodes, parent_node_dict


def draw_robot(frame, node, size=5, color=(0, 0, 255)):  # Assuming RED
    # Check if node contains theta; if not, use a default value of 0
    if len(node) == 3:
        x, y, theta = node
    else:
        x, y = node
        theta = 0  # Default orientation

    theta_rad = np.radians(theta)
    end_x = int(x + np.cos(theta_rad) * size * 2)
    end_y = int(y - np.sin(theta_rad) * size * 2)  # Negative due to y-axis inversion in images

    # Draw the robot
    cv2.circle(frame, (int(x), int(y)), size, color, -1)
    # Draw the orientation line
    cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), (0, 255, 255), 2, tipLength=0.5)

def visualize_path(canvas, path, explored_nodes, start, goal, parent_node_dict):
    print("Video generation started")
    height, width, _ = canvas.shape
    # Save animation as a video
    out = cv2.VideoWriter("optimal_path.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    video_start_time = time.time()

    # Draw explored nodes
    for node in explored_nodes:
        cv2.circle(canvas, (int(node[0]), int(node[1])), 1, (255, 0, 0), -1)

    # Draw the path from the goal to the start by backtracking using the parent_node_dict
    current_node = goal[:3]  # Assuming goal includes orientation
    while current_node[:2] != start[:2]:  # Compare only positions
        parent_node = parent_node_dict.get(current_node[:2])  # Retrieve parent using position
        if parent_node is None:
            break
        cv2.line(canvas, (int(current_node[0]), int(current_node[1])), (int(parent_node[0]), int(parent_node[1])), (0, 0, 0), 10)
        current_node = parent_node

    # Draw each step on the path with orientation
    for i in range(len(path)):
        frame = canvas.copy()
        node = path[i]
        cv2.circle(frame, (int(node[0]), int(node[1])), 5, (0, 255, 255), -1)  # Draw node
        draw_robot(frame, node)  # Draw robot orientation
        out.write(frame)  # Save frame after drawing the node and orientation
        cv2.waitKey(50)  # Small delay to animate the movement

    video_end_time = time.time()
    print(f"Video generation completed in {video_end_time - video_start_time:.2f} seconds")

    # cv2.imshow("A* Optimal Path", canvas)  # Show the final path
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    out.release()  # Close the video writer


def main():

    canvas = generate_map()
    clearance = float(input("Enter the clearance: "))
    step_size = float(input("Enter the step size: "))

    while True:
        start_x, start_y, start_theta = map(float, input(
            "Enter start x, y, and theta separated by commas (x, y, theta): ").split(','))
        if not is_obstacle(start_x, start_y, clearance) and \
                (0 <= start_x < canvas.shape[1]) and (0 <= start_y < canvas.shape[0]) and \
                (abs(start_theta) % 30 == 0):
            break
        print("Invalid start coordinates or theta not a multiple of 30. Please try again.")
    start = (start_x, start_y, start_theta)
    while True:
        goal_x, goal_y, goal_theta = map(float, input(
            "Enter goal x, y, and goal orientation separated by commas (x, y, theta): ").split(','))
        if not is_obstacle(goal_x, goal_y, clearance) and \
                (0 <= goal_x < canvas.shape[1]) and (0 <= goal_y < canvas.shape[0]) and \
                (abs(goal_theta) % 30 == 0):
            break
        print("Invalid goal coordinates or theta not a multiple of 30. Please try again.")
    goal = (goal_x, goal_y, goal_theta)



    #Alternate way to get user input

    # if is_obstacle(start_x, start_y,clearance) or is_obstacle(goal_x, goal_y,clearance):
    #
    #     print("Invalid start or goal coordinates. Please try again.")
    #     return

    # start_x, start_y, start_theta = map(float, input("Enter start x, y, and theta separated by commas (x, y, theta): ").split(','))
    # start = (start_x, start_y, start_theta)
    #
    # goal_x, goal_y, goal_theta = map(float, input("Enter goal x, y, and goal orientation separated by commas (x, y, theta): ").split(','))
    # goal = (goal_x, goal_y, goal_theta)

    start_time = time.time()
    path, explored_nodes, parent_node_dict = astar(start, goal, canvas, clearance, step_size)
    end_time = time.time()

    if not path:
        print("No path found between start and goal.")
    else:
        print(f"Path found in {end_time - start_time:.2f} seconds.")
        visualize_path(canvas, path, explored_nodes, start, goal, parent_node_dict)


if __name__ == "__main__":
    # cProfile.run('main()', 'profile_stats')
    # p = pstats.Stats('profile_stats')
    # p.sort_stats('cumulative').print_stats(10)
    main()

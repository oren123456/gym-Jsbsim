import os
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from threat import Threat
from scipy.spatial.distance import cdist


def get_neighbors(grid_size, point):
    neighbors = []
    x, y = point
    rows, cols = grid_size
    # Check left neighbor
    if x > 0:
        neighbors.append((x - 1, y))
    # Check right neighbor
    if x < cols - 1:
        neighbors.append((x + 1, y))
    # Check upper neighbor
    if y > 0:
        neighbors.append((x, y - 1))
    # Check lower neighbor
    if y < rows - 1:
        neighbors.append((x, y + 1))
    return neighbors


def create_threats(number_of_threats, grid_size, threat_radius_min, threat_radius_max):
    threats = []
    for _ in range(number_of_threats):
        center_coordinates = (random.randint(-int(grid_size/2), int(1.5*grid_size)), random.randint(-int(grid_size/2), int(1.5*grid_size)))
        # direction = (random.randint(-1, 1), random.randint(-1, 1))
        radius = random.randint(threat_radius_min, threat_radius_max)
        threats.append(Threat(center_coordinates, radius))
    return threats


def save_grid(grid, grd_path, colors_max_value):
    np.save(grd_path, grid)
    # Display the colored grid
    fig, ax = plt.subplots()
    cax = ax.imshow(grid, cmap='viridis', interpolation='nearest', vmax=colors_max_value)
    cbar = plt.colorbar(cax, label='Gradient Values')
    plt.close()
    # save figure
    img_path = f'{grd_path}.png'
    fig.savefig(img_path)


def create_scenario(grid_size=5, steps=1, number_of_threats=1, threat_radius_min=1, threat_radius_max=1):
    threats = create_threats(number_of_threats, grid_size, threat_radius_min, threat_radius_max)

    # Get the current date and time
    current_datetime = datetime.now()
    folder_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(os.getcwd() + '/scenarios', folder_name)
    os.makedirs(folder_path)

    # List to store frames
    frames = []
    num_frames = steps
    for i in range(steps):
        grid = np.zeros((grid_size, grid_size))
        for t in threats:
            grid = t.add_gradient_circle(grid)

        grd_path = f'{folder_path}/frame_{i}'
        save_grid(grid, grd_path, 500)
        frames.append(Image.open(f'{grd_path}.png'))

    # Save frames as a GIF
    frames[0].save(f'{folder_path}/scenario_animation.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)


def load_all_npy_files(directory_path):
    npy_files = [file for file in os.listdir(directory_path) if file.startswith('frame_') and file.endswith('.npy')]

    loaded_grids = []
    for npy_file in npy_files:
        file_path = os.path.join(directory_path, npy_file)
        loaded_grid = np.load(file_path)
        loaded_grids.append(loaded_grid)

    return loaded_grids


def get_points_at_distance(grid_size, given_point, distance):
    grid = np.zeros(grid_size)
    # Calculate distances from the given point to all grid points
    distances = cdist(np.array([given_point]), np.argwhere(grid == 0), metric='euclidean')
    # Find indices of points at distance 5
    indices_at_distance = np.argwhere(distances[0] <= distance)
    cols, rows = grid_size
    mapped_coordinates = []
    for location in indices_at_distance:
        row = location // cols
        col = location % cols
        if 0 <= row < rows and 0 <= col < cols:
            mapped_coordinates.append((row[0], col[0]))
    return mapped_coordinates


def create_cost_grids(folder_path):
    frames = []
    loaded_grids = load_all_npy_files(folder_path)
    grid_size_x, grid_size_y = loaded_grids[0].shape
    steps = len(loaded_grids)
    grid_scores = np.full((steps, grid_size_x, grid_size_y), np.inf)
    start_point = (0, 0)
    for i in range(steps):
        cells = get_points_at_distance((grid_size_x, grid_size_y), start_point, i)
        cost_grid_at_time_i = loaded_grids[i]
        for c in cells:
            x, y = c
            if i == 0:  # start cell
                c_score = 0
            else:
                c_score = grid_scores[i - 1, x, y]
            neighbors = get_neighbors(cost_grid_at_time_i.shape, c)
            for neighbor in neighbors:
                x, y = neighbor
                current_score = grid_scores[i, x, y]
                new_score = c_score + cost_grid_at_time_i[x, y] + 1
                if new_score < current_score:
                    grid_scores[i, x, y] = new_score

        grd_path = f'{folder_path}/score_{i}'
        save_grid(grid_scores[i], grd_path, 1500)
        frames.append(Image.open(f'{grd_path}.png'))

    frames[0].save(f'{folder_path}/score_animation.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)


if __name__ == "__main__":
    # num_of_scenarios = 1
    # for _ in range(num_of_scenarios):
    #     create_scenario(grid_size=50, steps=100, number_of_threats=25, threat_radius_min=1, threat_radius_max=20)
    f_path = 'C:/GitHub/gym-Jsbsim/Path Finder/scenarios/2024-02-05_18-42-59/'
    create_cost_grids(f_path)

    # print(loaded_grids)
    # print(grid_scores)
# loaded_grid = np.load(grd_path + '.npy')
# Show the plot
# plt.show()

# reconstructed_array = mpimg.imread(img_path)
#
#        arrays_equal = np.array_equal(array_100x100, reconstructed_array)
#        if arrays_equal:
#            print("The arrays are equal.")
#        else:
#            print("The arrays are not equal.")

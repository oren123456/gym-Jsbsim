import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from threat import Threat
from scipy.spatial.distance import cdist
import imageio
import numpy as np


def get_neighbors(grid_size, point):
    neighbors = [point]
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
        # Check upper-left neighbor
    if x > 0 and y > 0:
        neighbors.append((x - 1, y - 1))
        # Check upper-right neighbor
    if x < cols - 1 and y > 0:
        neighbors.append((x + 1, y - 1))
        # Check lower-left neighbor
    if x > 0 and y < rows - 1:
        neighbors.append((x - 1, y + 1))
        # Check lower-right neighbor
    if x < cols - 1 and y < rows - 1:
        neighbors.append((x + 1, y + 1))
    return neighbors


def create_threats(number_of_threats, grid_size, threat_radius_min, threat_radius_max):
    threats = []
    for _ in range(number_of_threats):
        center_coordinates = (random.randint(-int(grid_size / 2), int(1.5 * grid_size)), random.randint(-int(grid_size / 2), int(1.5 * grid_size)))
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
        grid = np.ones((grid_size, grid_size))
        for t in threats:
            grid = t.add_gradient_circle(grid)

        grd_path = f'{folder_path}/frame_{i}'
        save_grid(grid, grd_path, 500)
        frames.append(Image.open(f'{grd_path}.png'))

    # Save frames as a GIF
    frames[0].save(f'{folder_path}/scenario_animation.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)


def load_all_npy_files(directory_path, name):
    npy_files = [file for file in os.listdir(directory_path) if file.startswith(name) and file.endswith('.npy')]

    # Define a custom sorting function to extract the ending number
    def get_ending_number(filename):
        return int(filename.split('_')[-1].split('.')[0])

    # Sort the list of filenames based on the ending number
    sorted_npy_files = sorted(npy_files, key=get_ending_number)

    loaded_grids = []
    for npy_file in sorted_npy_files:
        file_path = os.path.join(directory_path, npy_file)
        loaded_grid = np.load(file_path)
        loaded_grids.append(loaded_grid)

    return loaded_grids


def get_points_at_distance(grid_size, given_point, distance):
    grid = np.zeros(grid_size)
    # Calculate distances from the given point to all grid points
    distances = cdist(np.array([given_point]), np.argwhere(grid == 0), metric='euclidean')
    indices_at_distance = np.argwhere(distances[0] <= distance)
    cols, rows = grid_size
    mapped_coordinates = []
    for location in indices_at_distance:
        row = location // cols
        col = location % cols
        if 0 <= row < rows and 0 <= col < cols:
            mapped_coordinates.append((row[0], col[0]))
    return mapped_coordinates


def merge_gifs_vertical(gif_path1, gif_path2, output_path):
    # Read the GIFs
    gif1 = imageio.get_reader(gif_path1)
    gif2 = imageio.get_reader(gif_path2)

    # Get the dimensions of the GIFs
    height1, width1 = gif1.get_data(0).shape[:2]
    height2, width2 = gif2.get_data(0).shape[:2]

    # Ensure the GIFs have the same width
    if width1 != width2:
        raise ValueError("The GIFs must have the same width.")

    # Create a new GIF writer for the output
    with imageio.get_writer(output_path, duration=gif1.get_meta_data()['duration'], loop=0) as writer:
        for i in range(min(len(gif1), len(gif2))):
            # Read frames
            frame1 = gif1.get_data(i)
            frame2 = gif2.get_data(i)

            # Concatenate frames vertically
            merged_frame = np.concatenate((frame1, frame2), axis=0)

            # Write the merged frame
            writer.append_data(merged_frame)


def create_cost_grids(folder_path):
    frames = []
    loaded_grids = load_all_npy_files(folder_path, 'frame_')
    grid_size_x, grid_size_y = loaded_grids[0].shape
    steps = len(loaded_grids)
    grid_scores = np.full((steps, grid_size_x, grid_size_y), np.inf)
    # cells = set()
    # cells.add((0, 0))
    grid_scores[0, 0, 0] = 0
    for i in range(1, steps):
        # current_cells = cells.copy()
        for k in range(grid_size_x):
            for j in range(grid_size_y):
                my_cost = grid_scores[i - 1][k, j]
                if my_cost == np.inf:
                    continue
                for neighbor in get_neighbors((grid_size_x, grid_size_y), (k, j)):
                    x, y = neighbor
                    neighbor_cost = loaded_grids[i][x, y]
                    if neighbor == (k, j):
                        neighbor_cost -= 1
                    grid_scores[i, x, y] = min(grid_scores[i, x, y], my_cost + neighbor_cost)

        grd_path = f'{folder_path}/score_{i}'
        save_grid(grid_scores[i], grd_path, 1500)

        frames.append(Image.open(f'{grd_path}.png'))

    frames[0].save(f'{folder_path}/score_animation.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)
    merge_gifs_vertical(f'{folder_path}/score_animation.gif', f'{folder_path}/scenario_animation.gif', f'{folder_path}/cost&threats.gif')


def find_lowest_cost_to_point(folder_path, point):
    scores = load_all_npy_files(folder_path, 'score_')
    min_cost = np.inf
    time = - 1
    x, y = point
    for i in range(len(scores)):
        cost = scores[i][x, y]
        if cost < min_cost:
            min_cost = cost
            time = i
    print(f'cost:{min_cost} at time:{time}')
    return time


def get_path_to_point(scores, point, time, grid_size):
    if time > 0:
        min_cost = np.inf
        prev_point = point
        time -= 1
        for neighbor in get_neighbors(grid_size, point):
            x, y = neighbor
            cost = scores[time][x, y]
            if cost < min_cost:
                min_cost = cost
                prev_point = neighbor
        print(f'prev_point:{prev_point} cost: {min_cost}')
        path = get_path_to_point(scores, prev_point, time, grid_size, )
        path.append(prev_point)
        return path
    return []


def find_path(folder_path, point, time):
    loaded_frames = load_all_npy_files(folder_path, 'frame_')
    loaded_scores = load_all_npy_files(folder_path, 'score_')

    anim_frames = []
    path = get_path_to_point(loaded_scores, point, time, loaded_scores[0].shape)
    # path.reverse()
    current_path = []
    for i in range(len(path)):
        current_path.append(path[i])
        for p in current_path:
            loaded_frames[i][p] = np.inf
        save_grid(loaded_frames[i], f'{f_path}/path_{i}', 100)
        anim_frames.append(Image.open(f'{f_path}/path_{i}.png'))
    anim_frames[0].save(f'{folder_path}/path_animation.gif', save_all=True, append_images=anim_frames[1:], duration=200, loop=0)

if __name__ == "__main__":
    # num_of_scenarios = 1
    # for _ in range(num_of_scenarios):
    #     create_scenario(grid_size=50, steps=100, number_of_threats=25, threat_radius_min=1, threat_radius_max=20)
    f_path = 'C:/GitHub/gym-Jsbsim/Path Finder/scenarios/2024-02-07_14-29-01/'
    create_cost_grids(f_path)
    # point1 = (49, 49)
    # find_path(f_path, point1, find_lowest_cost_to_point(f_path, point1))

    # print(loaded_grids)
    # print(grid_scores)
# loaded_grid = np.load(grd_path + '.npy')
# Show the plot
# plt.show()

# for cell in current_cells:
#     neighbors = get_neighbors((grid_size_x, grid_size_y), cell)
#     cells.update(neighbors)
# cells = get_points_at_distance((grid_size_x, grid_size_y), start_point, i)

# reconstructed_array = mpimg.imread(img_path)
#
#        arrays_equal = np.array_equal(array_100x100, reconstructed_array)
#        if arrays_equal:
#            print("The arrays are equal.")
#        else:
#            print("The arrays are not equal.")
# def create_cost_grids(folder_path):
#     frames = []
#     loaded_grids = load_all_npy_files(folder_path)
#     grid_size_x, grid_size_y = loaded_grids[0].shape
#     steps = len(loaded_grids)
#     grid_scores = np.full((steps, grid_size_x, grid_size_y), np.inf)
#     cells = set()
#     cells.add((0, 0))
#     for i in range(steps):
#         cerrent_cells = cells.copy()
#         for cell in cerrent_cells:
#             neighbors = get_neighbors((grid_size_x, grid_size_y), cell)
#             cells.update(neighbors)
#         # cells = get_points_at_distance((grid_size_x, grid_size_y), start_point, i)
#         cost_grid_at_time_i = loaded_grids[i]
#         for c in cells:
#             x, y = c
#             if i == 0:  # start cell
#                 c_score = 0
#             else:
#                 c_score = grid_scores[i - 1, x, y]
#             neighbors = get_neighbors(cost_grid_at_time_i.shape, c)
#             for neighbor in neighbors:
#                 x, y = neighbor
#                 current_score = grid_scores[i, x, y]
#                 new_score = c_score + cost_grid_at_time_i[x, y] + 1
#                 if new_score < current_score:
#                     grid_scores[i, x, y] = new_score
#
#         grd_path = f'{folder_path}/score_{i}'
#         save_grid(grid_scores[i], grd_path, 1500)
#         frames.append(Image.open(f'{grd_path}.png'))
#
#     frames[0].save(f'{folder_path}/score_animation.gif', save_all=True, append_images=frames[1:], duration=200, loop=0)
#     merge_gifs_vertical(f'{folder_path}/score_animation.gif', f'{folder_path}/scenario_animation.gif', f'{folder_path}/result.gif')
#

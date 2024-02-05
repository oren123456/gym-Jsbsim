import random

import numpy as np


class Threat(object):

    def __init__(self, start,  radius):
        self.center = start
        self.direction = (random.randint(-1, 1), random.randint(-1, 1))
        self.radius = radius

    def add_gradient_circle(self, data):
        # Extract the coordinates of the center
        if random.random() > 0.95:
            self.direction = (random.randint(-1, 1), random.randint(-1, 1))
        self.center = (self.direction[0] + self.center[0], self.direction[1] + self.center[1])
        # Create a grid of coordinates
        x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
        # Calculate distances from each point to the center
        center_x, center_y = self.center
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Calculate gradient values based on distance
        gradient_values = 100 * (1 - distances / self.radius)
        # Clip values to ensure they are in the range [0, 100]
        gradient_values = np.clip(gradient_values, 0, 100)
        # Update the values within the circular region
        data[distances <= self.radius] += gradient_values[distances <= self.radius]
        return data

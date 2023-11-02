import math


def calculate_hpr_difference(current_location, current_hpr, goal_location):
    # Calculate the differences in latitude, longitude, and altitude
    delta_latitude = math.radians(goal_location[0] - current_location[0])
    delta_longitude = math.radians(goal_location[1] - current_location[1])
    delta_altitude = goal_location[2] - current_location[2]

    # Calculate the direction angle to the goal
    goal_heading = math.atan2(delta_longitude, delta_latitude)

    # Convert the heading angle to degrees
    goal_heading_degrees = math.degrees(goal_heading)

    # Calculate the difference in heading
    heading_difference = goal_heading_degrees - current_hpr[0]

    # Normalize the heading difference to be between -180 and 180 degrees
    while heading_difference > 180:
        heading_difference -= 360
    while heading_difference < -180:
        heading_difference += 360

    # Calculate the difference in pitch (using altitude difference)
    pitch_difference = math.degrees(math.atan2(delta_altitude, math.sqrt(delta_latitude ** 2 + delta_longitude ** 2)))

    return heading_difference, pitch_difference


# Example usage
current_location = (0, 0, 900)  # Latitude, Longitude, Altitude
current_hpr = (0, 0, 0)  # Heading, Pitch, Roll (degrees)
goal_location = (0, 15000, 800)  # Latitude, Longitude, Altitude

heading_difference, pitch_difference = calculate_hpr_difference(current_location, current_hpr, goal_location)
print("Heading Difference:", heading_difference, "degrees")
print("Pitch Difference:", pitch_difference, "degrees")

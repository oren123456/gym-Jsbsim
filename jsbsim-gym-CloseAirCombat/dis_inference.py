import math

from jsbsim_gym.features import JSBSimFeatureExtractor
from jsbsim_gym.jsbsim_gym import JSBSimEnv
from stable_baselines3 import  PPO
import gymnasium as gym

import socket
import time

from io import BytesIO

from opendis.DataOutputStream import DataOutputStream
from opendis.dis7 import EntityStatePdu, Vector3Float, Vector3Double, EulerAngles
from opendis.PduFactory import createPdu
from opendis.RangeCoordinates import *

import tkinter as tk
import threading
import numpy as np

UDP_PORT = 3001
DESTINATION_ADDRESS = "127.0.0.1"

udpSocket_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpSocket_s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

udpSocket_r = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpSocket_r.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
udpSocket_r.bind(("", UDP_PORT))

gps = GPS()  # conversion helper

latest_loc = None


def move_forward(loc, step_size_in_meters):
    # Unpack geocentric coordinates and angles
    x, y, z, psi, theta, phi, _, _, _ = loc

    # Forward movement direction in ECEF
    dx = np.cos(theta) * np.cos(psi)
    dy = np.cos(theta) * np.sin(psi)
    dz = np.sin(theta)

    # Move forward
    new_x = x + step_size_in_meters * dx
    new_y = y + step_size_in_meters * dy
    new_z = z + step_size_in_meters * dz
    return new_x, new_y, new_z


def recv():
    data = udpSocket_r.recv(1024)  # buffer size in bytes
    pdu = createPdu(data)
    pduTypeName = pdu.__class__.__name__

    if pdu.pduType == 1:  # PduTypeDecoders.EntityStatePdu:
        if pdu.entityID.entityID != 44:
            loc = (pdu.entityLocation.x,
                   pdu.entityLocation.y,
                   pdu.entityLocation.z,
                   pdu.entityOrientation.psi,
                   pdu.entityOrientation.theta,
                   pdu.entityOrientation.phi,
                   pdu.entityLinearVelocity.x,
                   pdu.entityLinearVelocity.y,
                   pdu.entityLinearVelocity.z
                   )
            return loc
            # body = gps.ecef2llarpy(*loc)
            #
            # print("Received {}\n".format(pduTypeName)
            #       + " Id        : {}\n".format(pdu.entityID.entityID)
            #       + " Latitude  : {:.2f} degrees\n".format(rad2deg(body[0]))
            #       + " Longitude : {:.2f} degrees\n".format(rad2deg(body[1]))
            #       + " Altitude  : {:.0f} meters\n".format(body[2])
            #       + " Yaw       : {:.2f} degrees\n".format(rad2deg(body[3]))
            #       + " Pitch     : {:.2f} degrees\n".format(rad2deg(body[4]))
            #       + " Roll      : {:.2f} degrees\n".format(rad2deg(body[5]))
            #       )
    else:
        print("Received {}, {} bytes".format(pduTypeName, len(data)), flush=True)
        return None


def reset_env(simulation):
    # Unpack geocentric coordinates and angles
    _, _, _, psi, theta, phi, v_east_fps, v_north_fps, v_down_fps = latest_loc
    x, y, z = move_forward(latest_loc, 1000)
    body = gps.ecef2llarpy(x, y, z, psi, theta, phi)
    vel = math.sqrt(v_east_fps ** 2 + v_north_fps ** 2)
    default_condition = {
        "ic/long-gc-deg": rad2deg(body[1]),  # geodesic longitude [deg]
        "ic/lat-geod-deg": rad2deg(body[0]),  # geodesic latitude  [deg]
        "ic/h-sl-ft": body[2],  # altitude above mean sea level [ft]
        "ic/psi-true-deg": rad2deg(body[3]),  # initial (true) heading [deg] (0, 360)
        "ic/u-fps": vel,  # body frame x-axis velocity [ft/s]  (-2200, 2200)
        "ic/v-fps": 0.0,  # body frame y-axis velocity [ft/s]  (-2200, 2200)
        "ic/w-fps": 0.0,  # body frame z-axis velocity [ft/s]  (-2200, 2200)
        "ic/p-rad_sec": 0.0,  # roll rate  [rad/s]  (-2 * pi, 2 * pi)
        "ic/q-rad_sec": 0.0,  # pitch rate [rad/s]  (-2 * pi, 2 * pi)
        "ic/r-rad_sec": 0.0,  # yaw rate   [rad/s]  (-2 * pi, 2 * pi)
        "ic/roc-fpm": 0.0,  # initial rate of climb [ft/min]
        "ic/terrain-elevation-ft": 0,
        'propulsion/set-running': -1  # -1 refers to "All Engines"
    }
    for prop, value in default_condition.items():
        simulation.set_property_value(prop, value)
    simulation.run_ic()  # Initializes the sim from the initial condition object
    print("reset")


def send_entity_state_pdu(simulation):
    pdu = EntityStatePdu()
    pdu.entityID.entityID = 42
    pdu.entityID.siteID = 17
    pdu.entityID.applicationID = 23
    pdu.marking.setString('Igor3d')

    # Entity in Monterey, CA, USA facing North, no roll or pitch
    lon = simulation.get_property_value("position/long-gc-deg")
    lat = simulation.get_property_value("position/lat-geod-deg")
    alt = simulation.get_property_value("position/h-sl-ft") * 0.3048
    roll = simulation.get_property_value("attitude/roll-rad")
    pitch = simulation.get_property_value("attitude/pitch-rad")
    yaw = simulation.get_property_value("attitude/heading-true-rad")
    montereyLocation = gps.llarpy2ecef(deg2rad(lon),  # longitude (radians)
                                       deg2rad(lat),  # latitude (radians)
                                       alt,  # altitude (meters)
                                       roll,  # roll (radians)
                                       pitch,  # pitch (radians)
                                       yaw  # yaw (radians)
                                       )

    # Conversion factor from feet per second to meters per second
    pdu.entityLocation = Vector3Double(montereyLocation[0], montereyLocation[1], montereyLocation[2])
    pdu.entityOrientation = EulerAngles(montereyLocation[3], montereyLocation[4], montereyLocation[5])

    ft_to_m = 0.3048
    v_east_fps = simulation.get_property_value("velocities/v-east-fps") * ft_to_m
    v_north_fps = simulation.get_property_value("velocities/v-north-fps") * ft_to_m
    v_down_fps = simulation.get_property_value("velocities/v-down-fps") * ft_to_m
    pdu.entityLinearVelocity = Vector3Float(v_east_fps, v_north_fps, v_down_fps)

    memory_stream = BytesIO()
    output_stream = DataOutputStream(memory_stream)
    pdu.serialize(output_stream)
    data = memory_stream.getvalue()
    udpSocket_s.sendto(data, (DESTINATION_ADDRESS, UDP_PORT))
    # print(f"{montereyLocation}")


def update_labels(root):
    """Update labels with new values (dummy example, replace with real values)."""
    if latest_loc != None:
        x, y, z, psi, theta, phi, v_east_fps, v_north_fps, v_down_fps = latest_loc
        body = gps.ecef2llarpy(x, y, z, psi, theta, phi)
        # print("Latitude  : {:.2f} degrees\n".format(rad2deg(body[0]))
        #       + " Longitude : {:.2f} degrees\n".format(rad2deg(body[1]))
        #       + " Altitude  : {:.0f} meters\n".format(body[2])
        #       + " Yaw       : {:.2f} degrees\n".format(rad2deg(body[3]))
        #       + " Pitch     : {:.2f} degrees\n".format(rad2deg(body[4]))
        #       + " Roll      : {:.2f} degrees\n".format(rad2deg(body[5]))
        #       )
        speed = math.sqrt(v_east_fps ** 2 + v_north_fps ** 2)*0.3048
        # Update labels stored in root
        root.alt_label.config(text=f"altitude(f): {body[2]:.2f}")
        root.heading_label.config(text=f"heading(deg): {rad2deg(body[3]):.2f}")
        root.speed_label.config(text=f"Speed(m/s): {speed:.2f}")
    # Call this function again after 1 second
    root.after(500, lambda: update_labels(root))


def submit_id(root):
    print(int(root.id_entry_box.get()))  # Get input and convert to integer)


def submit_location(root):
    print(int(root.long_entry_box.get()))  # Get input and convert to integer)


def submit_turn_command(root, env):
    env.target_heading_deg = int(root.heading_entry_box.get())
    env.target_altitude_ft = int(root.alt_entry_box.get())
    env.target_velocities_u_mps = int(root.speed_entry_box.get())


# Tkinter GUI setup
def create_gui(sim_env):
    root = tk.Tk()
    root.title("Control Panel")
    root.geometry("400x250")

    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, anchor="w", pady=5, padx=5)
    reset_button = tk.Button(button_frame, text="Reset Env", command=lambda: reset_env())
    reset_button.pack(side=tk.LEFT)
    # **Frame for Labels**
    labels_frame = tk.Frame(root)
    labels_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.alt_label = tk.Label(labels_frame, text="altitude(f)", font=("Arial", 8))
    root.alt_label.pack(anchor="w")
    root.heading_label = tk.Label(labels_frame, text="heading(deg)", font=("Arial", 8))
    root.heading_label.pack(anchor="w")
    root.speed_label = tk.Label(labels_frame, text="Speed(m/s)", font=("Arial", 8))
    root.speed_label.pack(anchor="w")
    # **Entry Box for Track Entity ID (Inside a Frame)**
    id_frame = tk.Frame(root)
    id_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.input_label = tk.Label(id_frame, text="track entity id:", font=("Arial", 8))
    root.input_label.pack(side=tk.LEFT)
    root.id_entry_box = tk.Entry(id_frame, font=("Arial", 8), width=8)
    root.id_entry_box.pack(side=tk.LEFT, padx=5)
    submit_button = tk.Button(id_frame, text="Submit", font=("Arial", 8), command=lambda: submit_id(root))
    submit_button.pack(side=tk.RIGHT, padx=5)
    # # **Entry Box for Set Location**
    location_frame = tk.Frame(root)
    location_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.location_input_label = tk.Label(location_frame, text="set location: lan", font=("Arial", 8))
    root.location_input_label.pack(side=tk.LEFT)
    root.long_entry_box = tk.Entry(location_frame, font=("Arial", 8), width=8)
    root.long_entry_box.pack(side=tk.LEFT)
    root.long_input_label = tk.Label(location_frame, text="long", font=("Arial", 8))
    root.long_input_label.pack(side=tk.LEFT, padx=5)
    root.lat_entry_box = tk.Entry(location_frame, font=("Arial", 8), width=8)
    root.lat_entry_box.pack(side=tk.LEFT)
    submit_button = tk.Button(location_frame, text="Submit", font=("Arial", 8), command=lambda: submit_location(root))
    submit_button.pack(side=tk.RIGHT, padx=5)  # Add spacing
    # **Set heading , alt, speed**
    turn_frame = tk.Frame(root)
    turn_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.heading_input_label = tk.Label(turn_frame, text="heading(deg):", font=("Arial", 8))
    root.heading_input_label.pack(side=tk.LEFT)
    root.heading_entry_box = tk.Entry(turn_frame, font=("Arial", 8), width=5)
    root.heading_entry_box.pack(side=tk.LEFT, padx=5)
    root.alt_input_label = tk.Label(turn_frame, text="alt(f):", font=("Arial", 8))
    root.alt_input_label.pack(side=tk.LEFT)
    root.alt_entry_box = tk.Entry(turn_frame, font=("Arial", 8), width=5)
    root.alt_entry_box.pack(side=tk.LEFT, padx=5)
    root.speed_input_label = tk.Label(turn_frame, text="speed(m/s):", font=("Arial", 8))
    root.speed_input_label.pack(side=tk.LEFT)
    root.speed_entry_box = tk.Entry(turn_frame, font=("Arial", 8), width=5)
    root.speed_entry_box.pack(side=tk.LEFT, padx=5)
    submit_button = tk.Button(turn_frame, text="Submit", font=("Arial", 8), command=lambda: submit_turn_command(root, sim_env))
    submit_button.pack(side=tk.RIGHT)
    # **Button Next to Entry Box**
    input_frame = tk.Frame(root)
    input_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.input_label = tk.Label(input_frame, text="Enter an integer above.", font=("Arial", 12))
    root.input_label.pack()

    # Start updating labels
    update_labels(root)
    root.mainloop()

env = gym.make("JSBSim-v0")
models_dir = f"models/best_model"
model = PPO.load(models_dir, env, device="cpu")
print("Loaded model from " + models_dir)
obs, info = env.reset()
done = False
steps = 0
rewards_sum = 0
sim_env = env.unwrapped

# Run the GUI in a separate thread so the main loop continues
threading.Thread(target=create_gui, args=(sim_env,), daemon=True).start()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, _ , info = env.step(action)
    send_entity_state_pdu(sim_env.simulation)
    latest_loc = recv()
    time.sleep(1)
env.close()

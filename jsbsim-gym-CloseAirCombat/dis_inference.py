import math

from jsbsim_gym.features import JSBSimFeatureExtractor
from jsbsim_gym.jsbsim_gym import JSBSimEnv
from stable_baselines3 import PPO
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


def reset_env(sim_env):
    # Unpack geocentric coordinates and angles
    sim_env.reset()
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
    # print(montereyLocation[3])
    ft_to_m = 0.3048
    # print(rad2deg(simulation.get_property_value("attitude/heading-true-rad") ))
    # print(simulation.get_property_value("velocities/v-east-fps") * ft_to_m)
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
        lat, lon, alt, roll, pitch, yaw = gps.ecef2llarpy(x, y, z, psi, theta, phi)
        speed = math.sqrt(v_east_fps ** 2 + v_north_fps ** 2 + v_down_fps ** 2)

        root.lat_label.config(text=f"lat(deg): {rad2deg(lat):.4f}")
        root.long_label.config(text=f"long(deg): {rad2deg(lon):.4f}")
        root.alt_label.config(text=f"altitude(f): {alt * 3.28084:.2f}")
        root.heading_label.config(text=f"heading(deg): {rad2deg(yaw):.2f}")
        root.speed_label.config(text=f"Speed(m/s): {speed:.2f}")
    # Call this function again after 1 second
    root.after(500, lambda: update_labels(root))


def submit_id(root):
    print(int(root.id_entry_box.get()))  # Get input and convert to integer)


def is_valid_float(entry_value):
    try:
        float(entry_value)  # Try converting to float
        return True
    except ValueError:
        return False


def is_valid(value_type, box):
    val = None
    if value_type == int and box.get().isdigit():
        val = int(box.get())
    elif value_type == float and is_valid_float(box.get()):
        val = float(box.get())
    if val:
        box.configure(bg="white")
        return True
    else:
        box.configure(bg="red")
        return False


def submit_reset_heading_speed(root, sim_env):
    if is_valid(int, root.reset_heading_entry_box):
        sim_env.simulation.set_property_value("ic/psi-true-deg", int(root.reset_heading_entry_box.get()))
    if is_valid(int, root.reset_speed_entry_box):
        sim_env.simulation.set_property_value("ic/u-fps", int(root.reset_speed_entry_box.get())*3.28084)
        # print(f"set ic/u-fps to: {int(root.reset_speed_entry_box.get())}")
    sim_env.simulation.run_ic()


def submit_reset_location(root, sim_env):
    if is_valid(int, root.reset_alt_entry_box):
        sim_env.simulation.set_property_value("ic/h-sl-ft", int(root.reset_alt_entry_box.get()))
    if is_valid(float, root.reset_long_entry_box):
        sim_env.simulation.set_property_value("ic/long-gc-deg", float(root.reset_long_entry_box.get()))
    if is_valid(float, root.reset_lat_entry_box):
        sim_env.simulation.set_property_value("ic/lat-gc-deg", float(root.reset_lat_entry_box.get()))
    sim_env.simulation.run_ic()


def submit_turn_command(root, sim_env):
    if is_valid(int, root.target_heading_entry_box):
        sim_env.target_heading_deg = int(root.target_heading_entry_box.get())
    if is_valid(int, root.target_alt_entry_box):
        sim_env.target_altitude_ft = int(root.target_alt_entry_box.get())
    if is_valid(int, root.target_speed_entry_box):
        sim_env.target_velocities_u_mps = int(root.target_speed_entry_box.get())


# Tkinter GUI setup
def create_gui(sim_env):
    root = tk.Tk()
    root.title("Control Panel")
    root.geometry("500x250")

    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, anchor="w", pady=5, padx=5)
    reset_button = tk.Button(button_frame, text="Reset Env", command=lambda: reset_env(sim_env))
    reset_button.pack(side=tk.LEFT)
    # **Frame for Labels**
    location_labels_frame = tk.Frame(root)
    location_labels_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.lat_label = tk.Label(location_labels_frame, text="lat(deg)", font=("Arial", 8))
    root.lat_label.pack(side=tk.LEFT)
    root.long_label = tk.Label(location_labels_frame, text="long(deg)", font=("Arial", 8))
    root.long_label.pack(side=tk.LEFT)
    root.alt_label = tk.Label(location_labels_frame, text="altitude(f)", font=("Arial", 8))
    root.alt_label.pack(side=tk.LEFT)
    # **Frame for Labels**
    heading_labels_frame = tk.Frame(root)
    heading_labels_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.heading_label = tk.Label(heading_labels_frame, text="heading(deg)", font=("Arial", 8))
    root.heading_label.pack(side=tk.LEFT)
    root.speed_label = tk.Label(heading_labels_frame, text="Speed(m/s)", font=("Arial", 8))
    root.speed_label.pack(side=tk.LEFT)

    # **Entry Box for Track Entity ID (Inside a Frame)**
    id_frame = tk.Frame(root)
    id_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.input_label = tk.Label(id_frame, text="track entity id:", font=("Arial", 8))
    root.input_label.pack(side=tk.LEFT)
    root.id_entry_box = tk.Entry(id_frame, font=("Arial", 8), width=8)
    root.id_entry_box.pack(side=tk.LEFT, padx=5)
    target_submit_button = tk.Button(id_frame, text="Submit", font=("Arial", 8), command=lambda: submit_id(root))
    target_submit_button.pack(side=tk.RIGHT, padx=5)
    # # Reset Location**
    reset_location_frame = tk.Frame(root)
    reset_location_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.reset_location_input_label = tk.Label(reset_location_frame, text="Reset: lat(deg)", font=("Arial", 8))
    root.reset_location_input_label.pack(side=tk.LEFT)
    root.reset_long_entry_box = tk.Entry(reset_location_frame, font=("Arial", 8), width=8)
    root.reset_long_entry_box.insert(0, "120")
    root.reset_long_entry_box.pack(side=tk.LEFT)
    root.reset_long_input_label = tk.Label(reset_location_frame, text="long(deg)", font=("Arial", 8))
    root.reset_long_input_label.pack(side=tk.LEFT, padx=5)
    root.reset_lat_entry_box = tk.Entry(reset_location_frame, font=("Arial", 8), width=8)
    root.reset_lat_entry_box.pack(side=tk.LEFT)
    root.reset_lat_entry_box.insert(0, "60")
    root.reset_alt_input_label = tk.Label(reset_location_frame, text="alt(f)", font=("Arial", 8))
    root.reset_alt_input_label.pack(side=tk.LEFT, padx=5)
    root.reset_alt_entry_box = tk.Entry(reset_location_frame, font=("Arial", 8), width=8)
    root.reset_alt_entry_box.pack(side=tk.LEFT)
    root.reset_alt_entry_box.insert(0, "10000")
    reset_submit_button = tk.Button(reset_location_frame, text="Submit", font=("Arial", 8), command=lambda: submit_reset_location(root, sim_env))
    reset_submit_button.pack(side=tk.RIGHT, padx=5)  # Add spacing
    # # Reset Heading speed**
    reset_location_frame_2 = tk.Frame(root)
    reset_location_frame_2.pack(side=tk.TOP, anchor="w", padx=30, pady=5)
    root.reset_heading_input_label = tk.Label(reset_location_frame_2, text="heading(deg)", font=("Arial", 8))
    root.reset_heading_input_label.pack(side=tk.LEFT, padx=5)
    root.reset_heading_entry_box = tk.Entry(reset_location_frame_2, font=("Arial", 8), width=8)
    root.reset_heading_entry_box.pack(side=tk.LEFT)
    root.reset_heading_entry_box.insert(0, "90")
    root.reset_speed_input_label = tk.Label(reset_location_frame_2, text="speed(m/s)", font=("Arial", 8))
    root.reset_speed_input_label.pack(side=tk.LEFT, padx=5)
    root.reset_speed_entry_box = tk.Entry(reset_location_frame_2, font=("Arial", 8), width=8)
    root.reset_speed_entry_box.pack(side=tk.LEFT)
    root.reset_speed_entry_box.insert(0, "300")
    reset_submit_button = tk.Button(reset_location_frame_2, text="Submit", font=("Arial", 8), command=lambda: submit_reset_heading_speed(root, sim_env))
    reset_submit_button.pack(side=tk.RIGHT, padx=5)  # Add spacing
    # **target heading , alt, speed**
    target_turn_frame = tk.Frame(root)
    target_turn_frame.pack(side=tk.TOP, anchor="w", padx=5, pady=5)
    root.target_heading_input_label = tk.Label(target_turn_frame, text="Target: heading(deg): 45", font=("Arial", 8))
    root.target_heading_input_label.pack(side=tk.LEFT)
    root.target_heading_entry_box = tk.Entry(target_turn_frame, font=("Arial", 8), width=5)
    root.target_heading_entry_box.pack(side=tk.LEFT, padx=5)
    root.target_heading_entry_box.insert(0, "45")
    root.target_alt_input_label = tk.Label(target_turn_frame, text="alt(f):", font=("Arial", 8))
    root.target_alt_input_label.pack(side=tk.LEFT)
    root.target_alt_entry_box = tk.Entry(target_turn_frame, font=("Arial", 8), width=5)
    root.target_alt_entry_box.pack(side=tk.LEFT, padx=5)
    root.target_alt_entry_box.insert(0, "5000")
    root.target_speed_input_label = tk.Label(target_turn_frame, text="speed(m/s):", font=("Arial", 8))
    root.target_speed_input_label.pack(side=tk.LEFT)
    root.target_speed_entry_box = tk.Entry(target_turn_frame, font=("Arial", 8), width=5)
    root.target_speed_entry_box.pack(side=tk.LEFT, padx=5)
    root.target_speed_entry_box.insert(0, "200")
    target_submit_button = tk.Button(target_turn_frame, text="Submit", font=("Arial", 8), command=lambda: submit_turn_command(root, sim_env))
    target_submit_button.pack(side=tk.RIGHT)

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
_sim_env = env.unwrapped
_sim_env.simulation.set_property_value("reset", 0)

# Run the GUI in a separate thread so the main loop continues
threading.Thread(target=create_gui, args=(_sim_env,), daemon=True).start()
while True:
    action, _ = model.predict(obs, deterministic=True)
    action = [0,0,0,0]
    obs, rewards, done, _, info = env.step(action)
    send_entity_state_pdu(_sim_env.simulation)
    latest_loc = recv()
    env.render()
    time.sleep(1)
env.close()

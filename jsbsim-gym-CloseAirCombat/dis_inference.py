from jsbsim_gym.features import JSBSimFeatureExtractor
from jsbsim_gym.jsbsim_gym import JSBSimEnv
from stable_baselines3 import SAC, PPO
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
    """
    Moves 1 km forward based on the given geocentric location and orientation.

    Args:
    loc (tuple): (x, y, z, psi, theta, phi) where
                 x, y, z are ECEF coordinates,
                 psi is yaw (heading) in radians,
                 theta is pitch in radians.

    Returns:
    tuple: (new_x, new_y, new_z) - Updated geocentric position
    """
    # Unpack geocentric coordinates and angles
    x, y, z, psi, theta, phi = loc

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
        if pdu.entityID.entityID == 44:
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


def send_entityStatePdu(simulation):
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

    FT_TO_M = 0.3048
    v_east_fps = simulation.get_property_value("velocities/v-east-fps") * FT_TO_M
    v_north_fps = simulation.get_property_value("velocities/v-north-fps") * FT_TO_M
    v_down_fps = simulation.get_property_value("velocities/v-down-fps") * FT_TO_M
    pdu.entityLinearVelocity = Vector3Float(v_east_fps, v_north_fps, v_down_fps)

    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()
    udpSocket_s.sendto(data, (DESTINATION_ADDRESS, UDP_PORT))
    # print(f"{montereyLocation}")


# Tkinter GUI setup
def create_gui(simulation):
    root = tk.Tk()
    root.title("Control Panel")
    root.geometry("200x100")

    reset_button = tk.Button(root, text="Reset Env", command=lambda: reset_env(simulation))
    reset_button.pack(pady=20)

    root.mainloop()


env = gym.make("JSBSim-v0", )
RL_algo = "PPO"
models_dir = f"models/best_model"
model = PPO.load(models_dir, env)
print("Loaded model from " + models_dir)
vec_env = model.get_env()
obs = vec_env.reset()
done = False
# env.metadata["render_modes"] = ["rgb_array"]
steps = 0
rewards_sum = 0
sim = vec_env.get_attr("simulation")[0]

# Run the GUI in a separate thread so the main loop continues
threading.Thread(target=create_gui, args=(sim,), daemon=True).start()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = vec_env.step(action)
    send_entityStatePdu(sim)
    latest_loc = recv()
    time.sleep(1)

    # init_heading = np_random.uniform(0., 180.)
    # init_altitude = np_random.uniform(14000., 30000.)
    # init_velocities_u = np_random.uniform(400., 1200.)

print(f'Finished after {steps} steps. Reward: {rewards_sum}')
env.close()

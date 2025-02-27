#!python

__author__ = "DMcG"
__date__ = "$Jun 23, 2015 10:27:29 AM$"

import socket
import time

from io import BytesIO

from opendis.DataOutputStream import DataOutputStream
from opendis.dis7 import EntityStatePdu
from opendis.RangeCoordinates import *

from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import PPO
import gymnasium as gym

UDP_PORT = 3001
DESTINATION_ADDRESS = "127.0.0.1"

udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpSocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

gps = GPS()  # conversion helper

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0", )

RL_algo = "PPO"
models_dir = f"models/best_PPO_model"

model = PPO.load(models_dir, env)
print("Loaded model from " + models_dir)
vec_env = model.get_env()
obs = vec_env.reset()
done = False
# env.metadata["render_modes"] = ["rgb_array"]
steps = 0
rewards_sum = 0
while not done:
    # render_data = env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, info = vec_env.step(action)
    rewards_sum += rewards
    steps += 1
    env.render()
print(f'Finished after {steps} steps. Reward: {rewards_sum}')
env.close()


def send():
    pdu = EntityStatePdu()
    pdu.entityID.entityID = 42
    pdu.entityID.siteID = 17
    pdu.entityID.applicationID = 23
    pdu.marking.setString('Igor3d')
    pdu.exerciseID = 11

    # Entity in Monterey, CA, USA facing North, no roll or pitch
    montereyLocation = gps.llarpy2ecef(deg2rad(36.6),  # longitude (radians)
                                       deg2rad(-121.9),  # latitude (radians)
                                       1,  # altitude (meters)
                                       0,  # roll (radians)
                                       0,  # pitch (radians)
                                       0  # yaw (radians)
                                       )

    pdu.entityLocation.x = montereyLocation[0]
    pdu.entityLocation.y = montereyLocation[1]
    pdu.entityLocation.z = montereyLocation[2]
    pdu.entityOrientation.psi = montereyLocation[3]
    pdu.entityOrientation.theta = montereyLocation[4]
    pdu.entityOrientation.phi = montereyLocation[5]

    memoryStream = BytesIO()
    outputStream = DataOutputStream(memoryStream)
    pdu.serialize(outputStream)
    data = memoryStream.getvalue()

    while True:
        udpSocket.sendto(data, (DESTINATION_ADDRESS, UDP_PORT))
        print("Sent {}. {} bytes".format(pdu.__class__.__name__, len(data)))
        time.sleep(1)


send()

import math

import jsbsim
import gymnasium as gym
import numpy as np
from typing import Optional

from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
from .visualization.quaternion import Quaternion

# Initialize format for the environment state vector
STATE_FORMAT = [
    "position/lat-gc-rad",  # GetLatitude
    "position/long-gc-rad",  # GetLongitude
    "position/h-sl-meters",  # Returns the current altitude above sea level
    "velocities/mach",  # Gets the Mach number
    "aero/alpha-rad",  # angle of attack in radians
    "aero/beta-rad",  # Yaw Damper Beta
    "velocities/p-rad_sec",  # body frame angular velocity component.
    "velocities/q-rad_sec",  # body frame angular velocity component.
    "velocities/r-rad_sec",  # body frame angular velocity component.
    "attitude/phi-rad",  # Retrieves a vehicle Euler angle component
    "attitude/theta-rad",  # Retrieves a vehicle Euler angle component
    "attitude/psi-rad",  # Retrieves a vehicle Euler angle component
]

# PropertyManager->Tie("attitude/pitch-rad", this, (int)eTht, (PMF) & FGPropagate::GetEuler);
# PropertyManager->Tie("attitude/heading-true-rad", this, (int)ePsi, (PMF) & FGPropagate::GetEuler);

STATE_LOW = np.array([
    -np.inf,
    -np.inf,
    0,
    0,
    -np.pi,
    -np.pi,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.pi,
    -np.pi,
    -np.pi,
    -np.inf,
    -np.inf,
    0,
])

STATE_HIGH = np.array([
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.pi,
    np.pi,
    np.inf,
    np.inf,
    np.inf,
    np.pi,
    np.pi,
    np.pi,
    np.inf,
    np.inf,
    np.inf,
])

# Radius of the earth
RADIUS = 6.3781e6


class JSBSimEnv(gym.Env):
    """
    ### Description
    Gym environment using JSBSim to simulate an F-16 aerodynamics model with a
    simple point-to-point navigation task. The environment terminates when the
    agent enters a cylinder around the goal or crashes by flying lower than sea
    level. The goal is initialized at a random location in a cylinder around the
    agent's starting position. 

    ### Observation
    The observation is given as the position of the agent, velocity (mach, alpha,
    beta), angular rates, attitude, and position of the goal (concatenated in
    that order). Units are meters and radians. 

    ### Action Space
    Actions are given as normalized body rate commands and throttle command. 
    These are passed into a low-level PID controller built into the JSBSim model
    itself. The rate commands should be normalized between [-1, 1] and the 
    throttle command should be [0, 1].

    ### Rewards
    A positive reward is given for reaching the goal and a negative reward is 
    given for crashing. It is recommended to use the PositionReward wrapper 
    below to eliminate the problem of sparse rewards.
    """

    def __init__(self, root='.'):
        super().__init__()

        # Set observation and action space format
        self.observation_space = gym.spaces.Box(np.float32(STATE_LOW), STATE_HIGH, (15,))
        self.action_space = gym.spaces.Box(np.array([-1, -1, -1, 0]), 1, (4,))
        self.metadata["render_modes"] = ["rgb_array"]
        self.render_mode = "rgb_array"

        # Initialize JSBSim
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        #  JSBSIM default dT is 0.0083 ( 1/120 ) per step
        self.simulation.set_dt(0.1)

        # Load F-16 model and set initial conditions
        self.simulation.load_model('f16')
        self._set_initial_conditions()
        self.simulation.run_ic()

        self.down_sample = 1
        self.state = np.zeros(12)
        self.goal = np.zeros(3)
        self.dg = 400  # passing distance to goal - delta achieved goal
        self.viewer = None
        #
        # self.simulation.print_simulation_configuration()

        # TacView
        self._create_records = False

    def _set_initial_conditions(self):
        # Set engines running, forward velocity, and altitude
        self.simulation.set_property_value('propulsion/set-running', -1)  # -1 refers to "All Engines"
        self.simulation.set_property_value('ic/u-fps', 900.)  # Sets the initial body axis X velocity
        self.simulation.set_property_value('ic/h-sl-ft', 5000)  # Set the altitude SL ( sea level )

    def step(self, action):
        timestamp = self.simulation.get_sim_time()
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

        # Pass control inputs to JSBSim
        self.simulation.set_property_value("fcs/aileron-cmd-norm", roll_cmd)  # Sets the aileron command
        self.simulation.set_property_value("fcs/elevator-cmd-norm", pitch_cmd)  # Sets the elevator command
        self.simulation.set_property_value("fcs/rudder-cmd-norm", yaw_cmd)  # Sets the rudder command
        self.simulation.set_property_value("fcs/throttle-cmd-norm", throttle)  # Sets the throttle command

        # We take multiple steps of the simulation per step of the environment
        for _ in range(self.down_sample):
            # Freeze fuel consumption
            # self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000)
            # self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000)

            # Set gear up
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)  # gear command 0 for up
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)  # gear command 0 for up

            self.simulation.run()

        # Get the JSBSim state and save to self.state
        self._get_state()

        reward = 0
        done = False

        ep_info = {"goal": 0}
        # Check for collision with ground
        if self.state[2] < 10:
            ep_info["goal"] = -1
            reward = -100
            done = True

        # Check if reached goal
        if np.sqrt(np.sum((self.state[:2] - self.goal[:2]) ** 2)) < self.dg and abs(
                self.state[2] - self.goal[2]) < self.dg:
            ep_info["goal"] = 1
            reward = 100
            done = True

        # reward += self.simulation.get_property_value("propulsion/tank/contents-lbs")/10000
        # print(self.simulation.get_property_value("propulsion/tank/contents-lbs"))
        # self.simulation.set_property_value("propulsion/tank[1]/contents-lbs")

        return np.hstack([self.state, self.goal], dtype=np.float32), reward, done, False, ep_info

    def _get_state(self):
        # Gather all state properties from JSBSim
        for i, property in enumerate(STATE_FORMAT):
            self.state[i] = self.simulation.get_property_value(property)

        # Rough conversion to meters. This should be fine near zero lat/long
        self.state[:2] *= RADIUS

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.simulation.run_ic()  # Initializes the sim from the initial condition object
        self.simulation.set_property_value('propulsion/set-running', -1)  # -1 refers to "All Engines"
        # Set fuel consumption
        self.simulation.set_property_value("propulsion/tank/contents-lbs",
                                           1000)  # Fuel transfer first tank defined in f16.xml
        self.simulation.set_property_value("propulsion/tank[1]/contents-lbs",
                                           1000)  # Fuel transfer second tank defined in f16.xml

        # Generate a new goal
        rng = np.random.default_rng(seed)
        # distance = rng.random() * 9000 + 1000  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        distance = 9000
        bearing = rng.random() * 2 * np.pi
        altitude = rng.random() * 3000

        self.goal[:2] = np.cos(bearing), np.sin(bearing)
        self.goal[:2] *= distance
        self.goal[2] = altitude

        # Get state from JSBSim and save to self.state
        self._get_state()
        return np.hstack([self.state, self.goal], dtype=np.float32)

    def render(self):
        scale = 1e-3

        if self.viewer is None:
            self.viewer = Viewer(1280, 720)

            f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.f16 = RenderObject(f16_mesh)
            self.f16.transform.scale = 1 / 30
            self.f16.color = 0, 0, .4

            goal_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "cylinder.obj")
            self.cylinder = RenderObject(goal_mesh)
            self.cylinder.transform.scale = scale * 100
            self.cylinder.color = 0, .4, 0

            self.viewer.objects.append(self.f16)
            self.viewer.objects.append(self.cylinder)
            self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 21, 1.))

        # Rough conversion from lat/long to meters
        x, y, z = self.state[:3] * scale

        self.f16.transform.z = x
        self.f16.transform.x = -y
        self.f16.transform.y = z

        rot = Quaternion.from_euler(*self.state[9:])
        rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
        self.f16.transform.rotation = rot

        # self.viewer.set_view(-y , z + 1, x - 3, Quaternion.from_euler(np.pi/12, 0, 0, mode=1))

        x, y, z = self.goal * scale

        self.cylinder.transform.z = x
        self.cylinder.transform.x = -y
        self.cylinder.transform.y = z

        r = self.f16.transform.position - self.cylinder.transform.position
        rhat = r / np.linalg.norm(r)
        x, y, z = r
        yaw = np.arctan2(-x, -z)
        pitch = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))

        self.viewer.set_view(*(r + self.cylinder.transform.position + rhat + np.array([0, .33, 0])),
                             Quaternion.from_euler(-pitch, yaw, 0, mode=1))

        # print(self.f16.transform.position)

        # rot = Quaternion.from_euler(-self.state[10], -self.state[11], self.state[9], mode=1)

        self.viewer.render()

        render_modes = self.metadata.get("render_modes")
        if render_modes is not None:
            if 'rgb_array' in render_modes:
                return self.viewer.get_frame()

    # def tacview_log(self,  filepath='./JSBSimRecording.txt.acmi'):
    #     """Renders the environment.
    #     Note:
    #         Make sure that your class's metadata 'render.modes' key includes
    #           the list of supported modes. It's recommended to call super()
    #           in implementations to use the functionality of this method.
    #     :param mode: str, the mode to render with
    #     """
    #     if not self._create_records:
    #         with open(filepath, mode='w', encoding='utf-8-sig') as f:
    #             f.write("FileType=text/acmi/tacview\n")
    #             f.write("FileVersion=2.1\n")
    #             f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
    #         self._create_records = True
    #     with open(filepath, mode='a', encoding='utf-8-sig') as f:
    #         #timestamp = self.current_step * self.time_interval
    #         timestamp = self.simulation.get_sim_time()
    #         f.write(f"#{timestamp:.2f}\n")
    #         log_msg = sim.log()
    #             if log_msg is not None:
    #                 f.write(log_msg + "\n")
    #         for sim in self._tempsims.values():
    #             log_msg = sim.log()
    #             if log_msg is not None:
    #                 f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def calculate_hpr_difference(obs):
    # Calculate the differences in latitude, longitude, and altitude
    delta_latitude = math.radians(obs[-3] - obs[0])
    delta_longitude = math.radians(obs[-2] - obs[1])
    delta_altitude = obs[-1] - obs[2]

    # Calculate the direction angle to the goal
    goal_heading = math.atan2(delta_longitude, delta_latitude)

    # Calculate the difference in heading
    heading_difference_degrees = math.degrees(goal_heading - obs[9])

    # Normalize the heading difference to be between -180 and 180 degrees
    while heading_difference_degrees > 180:
        heading_difference_degrees -= 360
    while heading_difference_degrees < -180:
        heading_difference_degrees += 360

    # Calculate the difference in pitch (using altitude difference)
    pitch_difference_degrees = math.degrees(
        math.atan2(delta_altitude, math.sqrt(delta_latitude ** 2 + delta_longitude ** 2)))

    return heading_difference_degrees, pitch_difference_degrees


class PositionReward(gym.Wrapper):
    """
    This wrapper adds an additional reward to the JSBSimEnv. The agent is 
    rewarded based when moving closer to the goal and penalized when moving away.
    Staying at the same distance will result in no additional reward. The gain 
    may be set to weight the importance of this reward.
    """

    def __init__(self, env, gain):
        super().__init__(env)
        self.last_distance = None
        self.gain = gain

    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        displacement = obs[-3:] - obs[:3]
        distance = np.linalg.norm(displacement)
        reward += self.gain * (self.last_distance - distance)
        #print(obs)
       # h, p = calculate_hpr_difference(obs)
        #print(h)
        self.last_distance = distance
        info['distance'] = distance
        return obs, reward, done, False, info

    # def reset(self):
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs = super().reset()
        displacement = obs[-3:] - obs[:3]
        self.last_distance = np.linalg.norm(displacement)
        return obs, {}


# Create entry point to wrapped environment
def wrap_jsbsim(**kwargs):
    return PositionReward(JSBSimEnv(**kwargs), 1e-2)


# Register the wrapped environment
gym.register(
    id="JSBSim-v0",
    entry_point=wrap_jsbsim,
    max_episode_steps=1200
)

# Short example script to create and run the environment with
# constant action for 1 simulation second.
if __name__ == "__main__":
    from time import sleep

    env = JSBSimEnv()
    env.reset()
    env.render()
    for _ in range(300):
        env.step(np.array([0.05, -0.2, 0, .5]))
        env.render()
        sleep(1 / 30)
    env.close()

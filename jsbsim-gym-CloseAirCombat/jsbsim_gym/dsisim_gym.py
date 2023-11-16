import math

import jsbsim
import gymnasium as gym
import numpy as np
from typing import Optional

from .visualization.rendering import Viewer, load_mesh, load_shader, RenderObject, Grid
from .visualization.quaternion import Quaternion
from numpy.linalg import norm
from gymnasium.utils import seeding

import pymap3d

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
    -np.inf,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.inf,
    -np.inf,
])

STATE_HIGH = np.array([
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.inf,
    np.inf,
])

# Radius of the earth
RADIUS = 6.3781e6


def log_info(self, infos, total_num_steps):
    if self.use_wandb:
        wandb.log({k: v for k, v in infos.items()}, step=total_num_steps)
        # for k, v in infos.items():
        #     wandb.log({k: v}, step=total_num_steps)
    else:
        pass

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
        # self._geodetic = None
        # self.ic_psi_true_deg = 0
        self.target_heading_deg = 0
        self.target_altitude_ft = 0
        self.target_velocities_u_mps = 0
        self.current_step = 0
        self.heading_turn_counts = 0
        self.heading_check_time = 0
        self.acceleration_limit_x = 10.0
        self.acceleration_limit_y = 10.0
        self.acceleration_limit_z = 10.0
        self.altitude_limit = 2500
        self.max_steps = 2000
        self.max_heading_increment = 180  # degree
        self.max_altitude_increment = 7000  # feet
        self.max_velocities_u_increment = 100  # meter

        self.observation_space = gym.spaces.Box(np.float32(STATE_LOW), STATE_HIGH, (12,))
        # self.observation_space = gym.spaces.Box(low=-10, high=10., shape=(21,))
        self.action_space = gym.spaces.Box(-1, 1, (4,))
        # self.action_space = gym.spaces.Box(np.float32(STATE_LOW), STATE_HIGH, (15,))

        self.metadata["render_modes"] = ["rgb_array"]
        self.render_mode = "rgb_array"

        # Initialize JSBSim
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        #  JSBSIM default dT is 0.0083 ( 1/120 ) per step
        self.simulation.set_dt(1 / 60)

        self.state = np.zeros(10)

        # Load F-16 model and set initial conditions
        self.simulation.load_model('f16')
        self.reset()
        self.simulation.run_ic()

        self.down_sample = 1

        # self.goal = np.zeros(3)
        # self.viewer = None
        # self.simulation.print_simulation_configuration()

        # TacView
        self._create_records = False

        self.safe_altitude = 4.0
        self.danger_altitude = 3.5

        # self.lon0, self.lat0, self.alt0 = (120.0, 60.0, 0.0)

        self.np_random, seed = seeding.np_random(None)
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10
        self.check_interval = 30

    def get_extreme_state(self):
        extreme_velocity = self.simulation.get_property_value("velocities/eci-velocity-mag-fps") >= 1e10
        extreme_rotation = (
                norm([self.simulation.get_property_value("velocities/p-rad_sec"),
                      self.simulation.get_property_value("velocities/q-rad_sec"),
                      self.simulation.get_property_value("velocities/r-rad_sec"),
                      ]
                     ) >= 1000
        )
        extreme_altitude = self.simulation.get_property_value("position/h-sl-ft") >= 1e10
        extreme_acceleration = (
                max(
                    [
                        abs(self.simulation.get_property_value("accelerations/n-pilot-x-norm")),
                        abs(self.simulation.get_property_value("accelerations/n-pilot-y-norm")),
                        abs(self.simulation.get_property_value("accelerations/n-pilot-z-norm")),
                    ]
                ) > 1e1
        )  # acceleration larger than 10G
        # if extreme_altitude or extreme_rotation or extreme_velocity or extreme_acceleration:
        #     print("Continuing work on ")
        return extreme_altitude or extreme_rotation or extreme_velocity or extreme_acceleration

    def _judge_overload(self, ):
        flag_overload = False
        simulation_sim_time_sec = self.simulation.get_property_value("simulation/sim-time-sec")
        if simulation_sim_time_sec > 10:
            if math.fabs(self.simulation.get_property_value("accelerations/n-pilot-x-norm")) > self.acceleration_limit_x \
                    or math.fabs(self.simulation.get_property_value("accelerations/n-pilot-y-norm")) > self.acceleration_limit_y \
                    or math.fabs(self.simulation.get_property_value("accelerations/n-pilot-z-norm") + 1) > self.acceleration_limit_z:
                flag_overload = True
        return flag_overload

    def get_termination(self, ):
        # End up the simulation if the aircraft didn't reach the target heading in limited time.
        done = False
        # cur_step = info['current_step']
        # check heading when simulation_time exceed check_time
        simulation_sim_time_sec = self.simulation.get_property_value("simulation/sim-time-sec")
        if simulation_sim_time_sec >= self.heading_check_time:
            self.heading_check_time += self.check_interval
            if math.fabs(self.state[1]) > 10:  # delta_heading
                done = True
            # if current target heading is reached, random generate a new target heading
            else:
                delta = self.increment_size[self.heading_turn_counts]
                self.target_heading_deg += (self.np_random.uniform(-delta, delta) * self.max_heading_increment + 360) % 360
                self.target_altitude_ft += self.np_random.uniform(-delta, delta) * self.max_altitude_increment
                self.target_velocities_u_mps += self.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
                self.heading_turn_counts += 1
                # self.log(f'current_step:{cur_step} target_heading:{new_heading} '
                #          f'target_altitude_ft:{new_altitude} target_velocities_u_mps:{new_velocities_u}')

        return self.get_extreme_state() or self.current_step >= self.max_steps or \
            (self.simulation.get_property_value("position/h-sl-ft") * 0.3048) <= self.altitude_limit or \
            self._judge_overload() or done

    def get_reward(self, ):
        # Heading reward
        heading_r = math.exp(-((self.state[1] / 5.0) ** 2))  # heading_error_scale degrees
        alt_r = np.max(max(math.exp(-((self.state[0] / 15.24) ** 2)), 1e-200))  # alt_error_scale m
        roll_r = math.exp(-((self.state[4] / 0.35) ** 2))  # roll_error_scale radians ~= 20 degrees
        speed_r = math.exp(-((self.state[2] / 24) ** 2))  # speed_error_scale mps (~10%)
        reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)

        # Altitude reward
        ego_z = self.simulation.get_property_value("position/h-sl-ft") * 0.3048 / 1000  # unit: km
        ego_vz = self.simulation.get_property_value("velocities/v-down-fps") * 0.3048 / 340  # unit: mh
        pv = 0.
        if ego_z <= self.safe_altitude:
            pv = -np.clip(ego_vz / 0.2 * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 1.)
        ph = 0.
        if ego_z <= self.danger_altitude:
            ph = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.

        return reward + pv + ph

    def step(self, action):
        self.current_step += 1

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

        obs = self._get_state()
        reward = self.get_reward()
        done = self.get_termination()

        ep_info = {"goal": 0}
        # return np.hstack([self.state, self.goal], dtype=np.float32), reward, done, False#, ep_info
        return obs, reward, done, False, ep_info

    def _get_state(self):
        self.state[0] = (self.target_altitude_ft - self.simulation.get_property_value("position/h-sl-ft")) * 0.3048
        self.state[1] = self.target_heading_deg - self.simulation.get_property_value("attitude/psi-deg")
        self.state[2] = self.target_velocities_u_mps - self.simulation.get_property_value("velocities/u-fps") * 0.3048
        self.state[3] = self.simulation.get_property_value("position/h-sl-ft") * 0.3048
        self.state[4] = self.simulation.get_property_value("attitude/roll-rad")
        self.state[5] = self.simulation.get_property_value("attitude/pitch-rad")
        self.state[6] = self.simulation.get_property_value("velocities/u-fps") * 0.3048  # v_body_x   (unit: m/s)
        self.state[7] = self.simulation.get_property_value("velocities/v-fps") * 0.3048  # v_body_y   (unit: m/s)
        self.state[8] = self.simulation.get_property_value("velocities/w-fps") * 0.3048  # v_body_z   (unit: m/s)
        self.state[9] = self.simulation.get_property_value("velocities/vc-fps") * 0.3048  # vc        (unit: m/s)

        norm_obs = np.zeros(12, dtype=np.float32)
        norm_obs[0] = np.clip(self.state[0] / 1000, -1, 1)  # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = self.state[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        norm_obs[2] = self.state[2] / 340  # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = self.state[3] / 5000  # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(self.state[4])  # 4. ego_roll_sin
        norm_obs[5] = np.cos(self.state[4])  # 5. ego_roll_cos
        norm_obs[6] = np.sin(self.state[5])  # 6. ego_pitch_sin
        norm_obs[7] = np.cos(self.state[5])  # 7. ego_pitch_cos
        norm_obs[8] = self.state[6] / 340  # 8. ego_v_north    (unit: mh)
        norm_obs[9] = self.state[7] / 340  # 9. ego_v_east     (unit: mh)
        norm_obs[10] = self.state[8] / 340  # 10. ego_v_down    (unit: mh)
        norm_obs[11] = self.state[9] / 340  # 11. ego_vc        (unit: mh)
        return np.clip(norm_obs, self.observation_space.low, self.observation_space.high)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # super().reset() # seed=seed
        obs = super().reset()

        # Get state from JSBSim and save to self.state
        init_heading = self.np_random.uniform(0., 180.)
        init_altitude = self.np_random.uniform(14000., 30000.)
        init_velocities_u = self.np_random.uniform(400., 1200.)
        self.target_heading_deg = init_heading
        self.target_altitude_ft = init_altitude
        self.target_velocities_u_mps = init_velocities_u * 0.3048
        default_condition = {
            "ic/long-gc-deg": 120.0,  # geodesic longitude [deg]
            "ic/lat-geod-deg": 60.0,  # geodesic latitude  [deg]
            "ic/h-sl-ft": init_altitude,  # altitude above mean sea level [ft]
            "ic/psi-true-deg": init_heading,  # initial (true) heading [deg] (0, 360)
            "ic/u-fps": init_velocities_u,  # body frame x-axis velocity [ft/s]  (-2200, 2200)
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
            self.simulation.set_property_value(prop, value)
        self.simulation.run_ic()  # Initializes the sim from the initial condition object
        self.current_step = 0
        self.heading_turn_counts = 0

        return self._get_state(), {}

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some environments do not support rendering at all.) By convention,
        if mode is:
        - human: print on the terminal
        - txt: output to txt.acmi files
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """


        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval
                f.write(f"#{timestamp:.2f}\n")

                lon = self.simulation.get_property_value("position/long-gc-deg")
                lat = self.simulation.get_property_value("position/lat-geod-deg")
                alt = self.simulation.get_property_value("position/h-sl-ft") * 0.3048
                roll  =  self.simulation.get_property_value("attitude/roll-rad") * 180 / np.pi
                pitch = self.simulation.get_property_value("attitude/pitch-rad") * 180 / np.pi
                yaw = self.simulation.get_property_value("attitude/heading-true-rad") * 180 / np.pi
                log_msg = f"{1},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
                log_msg += f"Name=AAA,"
                log_msg += f"Color=Blue"
                if log_msg is not None:
                    f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


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
        # if reward < 0:
        #     print(reward)
        return obs, reward, done, False, info

    # def reset(self):
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # obs = super().reset()
        # displacement = obs[-3:] - obs[:3]
        # self.last_distance = np.linalg.norm(displacement)
        return super().reset()

    # Create entry point to wrapped environment


def wrap_jsbsim(**kwargs):
    return PositionReward(JSBSimEnv(**kwargs), 1e-2)

    # Register the wrapped environment


gym.register(
    id="JSBSim-v0",
    entry_point=wrap_jsbsim,
    max_episode_steps=1200
)
# return np.hstack([self.state, self.goal], dtype=np.float32)

# if done:
# self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}')
# info['heading_turn_counts'] = env.heading_turn_counts

# End up the simulation if the aircraft is on an extreme state.
# if done:
# env.agents[agent_id].crash()
# self.log(f'{agent_id} is on an extreme state! Total Steps={env.current_step}')

#   End up the simulation if acceleration are too high.
# if done:
#     env.agents[agent_id].crash()
#     self.log(f'{agent_id} acceleration is too high! Total Steps={env.current_step}')

# End up the simulation if altitude are too low.

# if done:
#     env.agents[agent_id].crash()
#     self.log(f'{agent_id} altitude is too low. Total Steps={env.current_step}')

# Terminate if max_step steps have passed
# if done:
#     self.log(f"{agent_id} step limits! Total Steps={env.current_step}")

# success = False
# if self.simulation.get_property_value("position/h-sl-ft") * 0.3048 <= self.altitude_limit:
#     print("aaaa")
# if   self.current_step >= self.max_steps:
#     print("aaaa")
# self._geodetic = [self.simulation.get_property_value("position/long-gc-deg"),
#                   self.simulation.get_property_value("position/lat-geod-deg"),
#                   self.simulation.get_property_value("position/h-sl-ft") * 0.3048]

# def render(self):
#     scale = 1e-3
#
#     if self.viewer is None:
#         self.viewer = Viewer(1280, 720)
#
#         f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
#         self.f16 = RenderObject(f16_mesh)
#         self.f16.transform.scale = 1 / 30
#         self.f16.color = 0, 0, .4
#
#         goal_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "cylinder.obj")
#         self.cylinder = RenderObject(goal_mesh)
#         self.cylinder.transform.scale = scale * 100
#         self.cylinder.color = 0, .4, 0
#
#         self.viewer.objects.append(self.f16)
#         self.viewer.objects.append(self.cylinder)
#         self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 21, 1.))
#
#     # Rough conversion from lat/long to meters
#     x, y, z = self.state[:3] * scale
#
#     self.f16.transform.z = x
#     self.f16.transform.x = -y
#     self.f16.transform.y = z
#
#     rot = Quaternion.from_euler(*self.state[9:])
#     rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
#     self.f16.transform.rotation = rot
#
#     # self.viewer.set_view(-y , z + 1, x - 3, Quaternion.from_euler(np.pi/12, 0, 0, mode=1))
#
#     x, y, z = self.goal * scale
#
#     self.cylinder.transform.z = x
#     self.cylinder.transform.x = -y
#     self.cylinder.transform.y = z
#
#     r = self.f16.transform.position - self.cylinder.transform.position
#     rhat = r / np.linalg.norm(r)
#     x, y, z = r
#     yaw = np.arctan2(-x, -z)
#     pitch = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))
#
#     self.viewer.set_view(*(r + self.cylinder.transform.position + rhat + np.array([0, .33, 0])),
#                          Quaternion.from_euler(-pitch, yaw, 0, mode=1))
#
#     # print(self.f16.transform.position)
#
#     # rot = Quaternion.from_euler(-self.state[10], -self.state[11], self.state[9], mode=1)
#
#     self.viewer.render()
#
#     render_modes = self.metadata.get("render_modes")
#     if render_modes is not None:
#         if 'rgb_array' in render_modes:
#             return self.viewer.get_frame()
#
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
# # TODO: real time rendering [Use FlightGear, etc.]


# Short example script to create and run the environment with
# constant action for 1 simulation second.
# if __name__ == "__main__":
#     from time import sleep
#
#     env = JSBSimEnv()
#     env.reset()
#     env.render()
#     for _ in range(300):
#         env.step(np.array([0.05, -0.2, 0, .5]))
#         env.render()
#         sleep(1 / 30)
#     env.close()

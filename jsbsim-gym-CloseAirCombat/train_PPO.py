import sys
import gymnasium
import jsbsim_gym.jsbsim_gym  # This line makes sure the environment is registered
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC, PPO, common
import time
import os
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from collections import deque
from stable_baselines3.common.utils import safe_mean
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import Logger, KVWriter, HumanOutputFormat
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from config import get_config

# from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import DummyVecEnv, ShareDummyVecEnv
import logging

from jsbsim_gym.jsbsim_gym import JSBSimEnv, PositionReward


class WandbOutputFormat(KVWriter):
    """   Dumps key/value pairs into TensorBoard's numeric format.   """

    def __init__(self, config: Dict = None, project: Optional[str] = None, name: Optional[str] = None):
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            # monitor_gym=True,  # auto-upload the videos of agents playing the game
            # save_code=True,  # optional
        )

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
        # print(f"Start:{step}")
        outputs = {}
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "tensorboard" in excluded:
                continue
            # print(f"{key}:{value}")
            outputs[key] = value
        wandb.log(outputs, step=step)

    def close(self) -> None:
        """
        closes the file
        """
        self.run.finish()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, models_dir: str, stats_window_size: int, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.models_dir = models_dir
        self.my_info_buffer = None  # type: Optional[deque]
        self._stats_window_size = stats_window_size
        # self.local_time = time.time()

        if self.my_info_buffer is None:
            self.my_info_buffer = deque(maxlen=self._stats_window_size)

    # def _init_callback(self) -> None:
    #     # Create folder if needed
    #     if self.save_path is not None:
    #         os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Add debug info at the end of each episode
        if "episode" in self.locals['env'].buf_infos[0]:
            info = self.locals['env'].buf_infos[0]
            self.my_info_buffer.extend([{"distance": info['distance'], "goal": info['goal']}])
            self.logger.record("oren/distance", safe_mean([ep_info["distance"] for ep_info in self.my_info_buffer]))
            self.logger.record("oren/goal", safe_mean([ep_info["goal"] for ep_info in self.my_info_buffer]))

        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            # x is time array & y is reward array
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward:{self.best_mean_reward:.2f}-Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                    self.model.save(self.models_dir)
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--episode-length', type=int, default=1000,
                       help="the max length of an episode")
    group.add_argument('--scenario-name', type=str, default='singlecombat_vsbaseline',
                       help="number of fighters controlled by RL policy")
    group.add_argument('--num-agents', type=int, default=1,
                       help="number of fighters controlled by RL policy")
    return parser.parse_known_args(args)[0]


# def make_render_env(all_args):
#     def get_env_fn(rank):
#         def init_env():
#             if all_args.env_name == "SingleCombat":
#                 env = SingleCombatEnv(all_args.scenario_name)
#             elif all_args.env_name == "SingleControl":
#                 env = SingleControlEnv(all_args.scenario_name)
#             elif all_args.env_name == "MultipleCombat":
#                 env = MultipleCombatEnv(all_args.scenario_name)
#             else:
#                 logging.error("Can not support the " + all_args.env_name + "environment.")
#                 raise NotImplementedError
#             env.seed(all_args.seed + rank * 1000)
#             return env
#         return init_env
#     if all_args.env_name == "MultipleCombat":
#         return ShareDummyVecEnv([get_env_fn(0)])
#     else:
#         # return DummyVecEnv([get_env_fn(0)])
#         return get_env_fn(0)

# print("Sample Action:", env.action_space.sample())
# print("Observation Space:", env.observation_space.shape)
# print("Observation Sample:", env.observation_space.sample())

policy_type = "PPO"
stats_window_size = 100
# Everything in the config dict is saved to wandb
config = {
    "total_timesteps": 2500000,
    "env_name": "JSBSim-v0",
}

# policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor, )

log_dir = f"logs/" + datetime.now().strftime("%H_%M_%S")
os.makedirs(log_dir, exist_ok=True)
#
env = PositionReward(JSBSimEnv(), 1e-2)
env = Monitor(env, log_dir)

env_checker.check_env(env)

parser = get_config()
all_args = parse_args(sys.argv[1:], parser)
# env = SingleControlEnv(all_args.scenario_name)  # make_render_env(all_args)
# env = Monitor(env, log_dir)
# num_agents = all_args.num_agents

models_dir = f"models/best_" + policy_type + "_model"
# if os.path.exists(models_dir + ".zip"):
#     print("Continuing work on " + models_dir)
#     if policy_type == "SAC":
#         model = SAC.load(models_dir, env, verbose=1, tensorboard_log=log_dir, policy_kwargs=policy_kwargs,
#                          gradient_steps=-1, device='auto', )
#     if policy_type == "PPO":
#         model = PPO.load(models_dir, env, verbose=1,  policy_kwargs=policy_kwargs,
#                          gradient_steps=-1, device='cpu', )
# else:
#     print("Creating a new model")
#     if policy_type == "SAC":
#         model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, device='auto')
#     if policy_type == "PPO":

#policy_kwargs=policy_kwargs,
model = PPO('MlpPolicy', env, verbose=1,  tensorboard_log=log_dir, device='cpu')
# learning_rate=linear_schedule(0.001)

# model._stats_window_size = stats_window_size
callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir, models_dir=models_dir,
                                            stats_window_size=stats_window_size)

logger = Logger(folder=log_dir, output_formats=[WandbOutputFormat(name=datetime.now().strftime("%H:%M:%S"),  project="FlyToPoint", config=config),
                                                HumanOutputFormat(sys.stdout)])
model.set_logger(logger)

# for i in range(1, 2):
model.learn(total_timesteps=int(config["total_timesteps"]), )  # callback=[callback, WandbCallback()]
# logger.close()
env.close()

# Usefull
# local_time = time.localtime(time.time())
# run_name = f"{str(local_time.tm_mon)}_{str(local_time.tm_mday)}_{str(local_time.tm_year)}-{str(local_time.tm_hour)}_" \
#            f"{str(local_time.tm_min)}_{str(local_time.tm_sec)}"
# for i in range(1,100):
#     model.learn(total_timesteps=TIMESTPES,reset_num_timesteps=False, tb_log_name=f"SAC-{int(time.time())}")
# mean_params = model.get_parameters()
# print("Sample Observation: ", env.observation_space.sample())
# print("Sample Action: ", env.action_space.sample())
# model.save(f"{models_dir}/{TIMESTPES*i}")
# model.save("models/jsbsim_sac")
# model.save_replay_buffer("models/jsbsim_sac_buffer")
# print("Observation Space: ", env.observation_space.shape)

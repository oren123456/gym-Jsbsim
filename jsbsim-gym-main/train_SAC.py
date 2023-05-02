import gym
import jsbsim_gym.jsbsim_gym  # This line makes sure the environment is registered
from os import path
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC, PPO
import time
import os
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, models_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.models_dir = models_dir

    # def _init_callback(self) -> None:
    #     # Create folder if needed
    #     if self.save_path is not None:
    #         os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

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


# print("Sample Action:", env.action_space.sample())
# print("Observation Space:", env.observation_space.shape)
# print("Observation Sample:", env.observation_space.sample())

models_dir = f"models/best_SAC_model"
log_dir = f"logs/{int(time.time())}-SAC"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("JSBSim-v0")
env = Monitor(env, log_dir)

# log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')

# try:
# model = PPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, device='auto',
#             learning_rate=linear_schedule(0.001), )
# model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, gradient_steps=-1,
#             device='auto')
if os.path.exists(models_dir + ".zip"):
    print("Continuing work on " + models_dir)
    model = SAC.load(models_dir, env, verbose=1, tensorboard_log=log_dir, policy_kwargs=policy_kwargs,
                     gradient_steps=-1, device='auto')
else:
    print("Creating a new model")
    model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, device='auto', )

callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir, models_dir=models_dir)

# model.learn(10000)

TIMESTPES = 3000000
# for i in range(1,100):
#     model.learn(total_timesteps=TIMESTPES,reset_num_timesteps=False, tb_log_name=f"SAC-{int(time.time())}")
model.learn(total_timesteps=int(TIMESTPES), callback=callback)
#     model.save(f"{models_dir}/{TIMESTPES*i}")
# finally:
# model.save("models/jsbsim_sac")
# model.save_replay_buffer("models/jsbsim_sac_buffer")

from stable_baselines3 import PPO
import os
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
from jsbsim_gym.features import JSBSimFeatureExtractor
import torch


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, best_model_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.best_model_dir = best_model_dir

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.best_model_dir}")
                    self.model.save(self.best_model_dir)
                    self.model.save(self.log_dir)

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
        # print(f'learning rate:{progress_remaining * initial_value}')
        return progress_remaining * initial_value

    return func


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    print(f'cuda.is_available: {torch.cuda.is_available()}')

    stats_window_size = 100
    log_dir = os.path.join('logs', datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))

    # set_random_seed(5)
    num_of_envs = 10
    vec_env = make_vec_env("JSBSim-v0", n_envs=num_of_envs, seed=5, vec_env_cls=SubprocVecEnv, monitor_dir=log_dir)
    # vec_env = VecFrameStack(vec_env, n_stack=num_of_envs)
    # vec_env.reset()

    # policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor, ) policy_kwargs=policy_kwargs,
    models_dir = f"models"
    if not os.path.exists(models_dir + "/best_model.zip"):
        print("Continuing work on " + models_dir)
        model = PPO.load(models_dir, vec_env, verbose=1, tensorboard_log=log_dir, device='cpu')  # , learning_rate=linear_schedule(0.0001)
    else:
        print("Creating a new model")
        model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=log_dir, device='cpu')  # , learning_rate=linear_schedule(0.0001)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, best_model_dir=models_dir)

    eval_callback = EvalCallback(vec_env, best_model_save_path=models_dir,
                                 log_path=log_dir, eval_freq=max(5000 // num_of_envs, 1),
                                 n_eval_episodes=5, deterministic=True,
                                 render=True)



    timesteps = 1e5
    model.learn(total_timesteps=int(timesteps), progress_bar=True, callback=eval_callback)
    # logger.close() WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2)
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "JSBSIM Basic Fly ")
    plt.show()
    vec_env.close()

    #
    # class WandbOutputFormat(KVWriter):
    #     """   Dumps key/value pairs into TensorBoard's numeric format.   """
    #
    #     def __init__(self, config: Dict = None, project: Optional[str] = None, name: Optional[str] = None):
    #         self.run = wandb.init(
    #             project=project,
    #             name=name,
    #             config=config,
    #             # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #             # monitor_gym=True,  # auto-upload the videos of agents playing the game
    #             # save_code=True,  # optional
    #         )
    #
    #     def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
    #         # print(f"Start:{step}")
    #         outputs = {}
    #         for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
    #             if excluded is not None and "tensorboard" in excluded:
    #                 continue
    #             # print(f"{key}:{value}")
    #             outputs[key] = value
    #         wandb.log(outputs, step=step)
    #
    #     def close(self) -> None:
    #         """
    #         closes the file
    #         """
    #         self.run.finish()

    # print(vec_env.action_space.sample())

    # Everything in the config dict is saved to wandb
    # config = {
    #     "total_timesteps": 2500000,
    #     "env_name": "JSBSim-v0",
    # }

    # def _init_callback(self) -> None:
    #     # Create folder if needed
    #     if self.save_path is not None:
    #         os.makedirs(self.save_path, exist_ok=True)

    # logger = Logger(folder=log_dir, output_formats=[WandbOutputFormat(name=datetime.now().strftime("%H:%M:%S"), project="FlyToPoint", config=config),
    #                                                 HumanOutputFormat(sys.stdout)])
    # logger = Logger(folder=log_dir, output_formats=[HumanOutputFormat(sys.stdout)])
    # model.set_logger(logger)

    # vec_env = MaxAndSkipEnv(vec_env, 4)
    # np.random.randint(0, 2 ** 31 - 1)

    # if os.path.exists(models_dir + ".zip"):
    #     env = gym.make("JSBSim-v0", )
    #     best_model = PPO.load(models_dir, env)
    #     model_vec_env = best_model.get_env()
    #     obs = model_vec_env.reset()
    #     done = False
    #     rewards_sum = 0
    #     while not done:
    #         action, _ = best_model.predict(obs, deterministic=True)
    #         obs, rewards, done, info = model_vec_env.step(action)
    #         rewards_sum += rewards
    #     self.best_mean_reward = np.mean(rewards_sum)

    # run.finish()
# run = wandb.init(
#     project="sb3",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     # monitor_gym=True,  # auto-upload the videos of agents playing the game
#     save_code=True,  # optional
# )

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

#
# def parse_args(args, parser):
#     group = parser.add_argument_group("JSBSim Env parameters")
#     group.add_argument('--episode-length', type=int, default=1000,
#                        help="the max length of an episode")
#     group.add_argument('--scenario-name', type=str, default='singlecombat_vsbaseline',
#                        help="number of fighters controlled by RL policy")
#     group.add_argument('--num-agents', type=int, default=1,
#                        help="number of fighters controlled by RL policy")
#     return parser.parse_known_args(args)[0]

# parser = get_config()
# all_args = parse_args(sys.argv[1:], parser)


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

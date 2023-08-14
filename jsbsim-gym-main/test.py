import jsbsim_gym.jsbsim_gym  # This line makes sure the environment is registered
from jsbsim_gym.features import JSBSimFeatureExtractor
from jsbsim_gym.jsbsim_gym import JSBSimEnv
from stable_baselines3 import SAC, PPO
import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0", )

RL_algo = "PPO"
# RL_algo= "SAC"

models_dir = f"models/best_" + RL_algo + "_model"

if RL_algo == "PPO":
    model = PPO.load(models_dir, env)
else:
    model = SAC.load(models_dir, env)

video_recorder = VideoRecorder(env, "video_" + RL_algo + ".mp4", enabled=True)

vec_env = model.get_env()
obs = vec_env.reset()
# obs = env.reset()
done = False
step = 0
env.metadata["render_modes"] = ["rgb_array"]
while not done:
    render_data = env.render()
    if step % 3 == 0:
        video_recorder.capture_frame()
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, info = vec_env.step(action)
    env.render()
    step += 1

video_recorder.close()

env.close()

# env = TimeLimit(env, max_episode_steps=10000)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# mp4_writer = iio.get_writer("video_" + RL_algo + ".mp4", format="ffmpeg", fps=30)
# render_data = env.render(mode='rgb_array')
# mp4_writer.append_data(render_data)
# gif_writer.append_data(render_data[::2,::2,:])

# mp4_writer.close()
# gif_writer.close()
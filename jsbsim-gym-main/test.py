import gym
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
import imageio as iio
from os import path
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC, PPO
from gym.wrappers import TimeLimit

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0", )
# env = TimeLimit(env, max_episode_steps=10000)

models_dir = f"models/best_PPO_model"
# models_dir = f"models/best_SAC_model"
model = PPO.load(models_dir, env)


mp4_writer = iio.get_writer("video.mp4", format="ffmpeg", fps=30)
gif_writer = iio.get_writer("video.gif", format="gif", fps=5)
obs = env.reset()
done = False
step = 0
while not done:
    render_data = env.render(mode='rgb_array')
    mp4_writer.append_data(render_data)
    if step % 6 == 0:
        gif_writer.append_data(render_data[::2,::2,:])

    action, _ = model.predict(obs, deterministic=True)
    # obs, _, done, _ = env.step(action)
    obs, reward, done, info = env.step(action)
    # print(reward)
    step += 1
mp4_writer.close()
gif_writer.close()
env.close()
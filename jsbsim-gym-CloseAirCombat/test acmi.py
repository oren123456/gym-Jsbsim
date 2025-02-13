from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize

# policy_kwargs = dict(
#     features_extractor_class=JSBSimFeatureExtractor
# )

env = gym.make("JSBSim-v0", )
# env.metadata["render_modes"] = ["human"]

models_dir = f"models/best_model"
model = PPO.load(models_dir, env, device="cpu",)
print("Loaded model from " + models_dir)
obs, info = env.reset()
done = False
steps = 0
rewards_sum = 0
while steps<10000:
    # render_data = env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, _ , info = env.step(action)
    if done:
        if env.unwrapped.simulation.get_property_value("position/h-sl-ft") * 0.3048 <= 1000:
            print("reset")
            env.reset()
    rewards_sum += rewards
    steps += 1
    env.render()
print(f'Finished after {steps} steps. Reward: {rewards_sum}')
env.close()

# env = TimeLimit(env, max_episode_steps=10000)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# mp4_writer = iio.get_writer("video_" + RL_algo + ".mp4", format="ffmpeg", fps=30)
# render_data = env.render(mode='rgb_array')
# mp4_writer.append_data(render_data)
# gif_writer.append_data(render_data[::2,::2,:])

# mp4_writer.close()
# gif_writer.close()
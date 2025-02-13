from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import PPO
import gymnasium as gym

policy_kwargs = dict(
    features_extractor_class=JSBSimFeatureExtractor
)

env = gym.make("JSBSim-v0", )

RL_algo = "PPO"
# RL_algo= "SAC"

models_dir = f"models/best_model"

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

# env = TimeLimit(env, max_episode_steps=10000)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# mp4_writer = iio.get_writer("video_" + RL_algo + ".mp4", format="ffmpeg", fps=30)
# render_data = env.render(mode='rgb_array')
# mp4_writer.append_data(render_data)
# gif_writer.append_data(render_data[::2,::2,:])

# mp4_writer.close()
# gif_writer.close()
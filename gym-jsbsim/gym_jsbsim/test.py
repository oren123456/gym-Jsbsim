import gym
import numpy as np
import gym_jsbsim
# from gym_jsbsim import C172P

# Create the JSBSim environment

# env = gym.make('JSBSim-TurnHeadingControlTask-F15-Shaping.EXTRA-FG')
env = gym.make('JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0')

# Set the random seed for reproducibility
np.random.seed(0)

# Reset the environment to its initial state
state = env.reset()

# Define the number of episodes and steps per episode
num_episodes = 100
num_steps = 1000

# Loop over episodes
for i in range(num_episodes):

    # Loop over steps
    for j in range(num_steps):

        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take a step in the environment using the sampled action
        next_state, reward, done, info = env.step(action)

        # Update the current state
        state = next_state

        # If the episode is finished, print the total reward and break out of the loop
        if done:
            print(f"Episode {i}: Total reward = {info['episode']['r']}")
            break

import gym
import d4rl # Import required to register environments

# Create the environment
env = gym.make('maze2d-umaze-v1')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations'].shape) # (N, dim_observation) numpy array of observations
print(dataset['actions'].shape) # (N, dim_action)
print(dataset['rewards'].shape) # (N, )
print(dataset['terminals'].shape) # (N, )
# print(dataset['actions'][0])
# print(dataset['actions'][1])
# print(dataset['actions'][999998])
# print(dataset['actions'][999999])


# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
#dataset = d4rl.qlearning_dataset(env)
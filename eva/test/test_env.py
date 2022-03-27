import gym
from gym import spaces
import os
import numpy as np
import d4rl

def test_env(env_id):
    env = gym.make(env_id)

    for episode in range(10):
        print("***********************************")
        print('Episode: {}'.format(episode))
        env.reset()
        
        for i in range(500):  # max steps per episode
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            #print(state)
            #print(reward)
            if done:
                break
                    
        print('Episode {} finished after {} timesteps.'.format(episode,i+1))
    
    print("***********************************")   
    print("Action space: %s"%(env.action_space)) 
    # action is continuous
    if isinstance(env.action_space, spaces.Box):
        act_dim = int(np.prod(env.action_space.shape))
    else:
        act_dim = env.action_space.n
    print("Action number: %d"%(act_dim))
        
    print("State space: %s"%(env.observation_space)) 
    # assume state is continuous
    if isinstance(env.observation_space, spaces.Box):
        state_dim = int(np.prod(env.observation_space.shape))
        print("State dimension: %d"%(state_dim)) 
    print("***********************************")
    
    env.close() 

if __name__ == "__main__":  
    # ---------------- gym ---------------
    # 'LunarLander-v2'  # action: 4(d), state: 8(c)
    # 'InvertedDoublePendulum-v2' # action: 1(c), state: 11(c)
    # 'Swimmer-v2' # action: 2(c), state: 8(c)
    # 'InvertedDoublePendulum-v2' # action: 1(c), state: 11(c)
    # ---------------- d4rl ---------------
    # 'minigrid-fourrooms-v0' # action: 7(d), state: 7*7*3(c)
    # 'maze2d-umaze-v1' # action: 2(c), state: 4(c)
    # 'antmaze-umaze-v0' # action: 8(c), state: 29(c)
    # 'kitchen-complete-v0' # action: 9(c), state: 60(c)
    test_env(env_id='LunarLander-v2')
import gym
from gym import spaces
import os
import numpy as np
import d4rl
from eva.envs.common import *
from eva.envs.gcsl_envs import goal_env
from eva.envs.mazelab.env import EmptyMazeEnv, UMazeEnv, FourRoomEnv
from eva.envs.bitflip import BitFlippingGymEnv

def test_env(env_type, env_id, fixed_start, fixed_goal):
    env = create_env(env_type, env_id, fixed_start=fixed_start, fixed_goal=fixed_goal)

    for episode in range(10):
        print("***********************************")
        print('Episode: {}'.format(episode))
        state = env.reset()
        if isinstance(env, goal_env.GoalEnv):
            #print(state)
            print("Observation: %s"%env.observation(state))
            print("Achieved goal: %s"%env.extract_goal(state))
            print("State: %s"%state)
        
        for i in range(500):  # max steps per episode
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            #print(state)
            print(reward)
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
    print("--------------------------------------")    
    print("Observation space: %s"%(env.observation_space)) 
    # assume state is continuous
    if isinstance(env.observation_space, spaces.Box):
        state_dim = int(np.prod(env.observation_space.shape))
        print("Observation dimension: %d"%(state_dim)) 
    print("--------------------------------------")
    if env_type == "gcsl":
        print("Goal space: %s"%(env.goal_space))   
        if isinstance(env.goal_space, spaces.Box):
            goal_dim = int(np.prod(env.goal_space.shape))
        print("Goal dimension: %d"%(goal_dim))
        #print("Sampled goal: %s"%(sample_goal(env)))
    print("--------------------------------------")
    #print(env.spec)
    if hasattr(env.spec, "max_episode_steps"):
        print("Env timestep limit: %d"%env.spec.max_episode_steps)
    print("***********************************")    
    
    env.close() 

def test_gym_env():
    env = BitFlippingGymEnv()
    #env = gym.make('FetchPickAndPlace-v1')
    #env = gym.make('FetchPush-v1')   
    #env = gym.make('FetchReach-v1')  
    #env = gym.make('FetchSlide-v1')
    #env = FourRoomEnv(start=(1,1), goal=(6,4), random_start=True, random_goal=True)

    #print(inspect.getmembers(BitFlippingEnvironment, predicate=inspect.isfunction))
    #env = gym.make("bit-flipping-v0")
    for episode in range(20):
        print("***********************************")
        print('Episode: {}'.format(episode))
        step = 0
        obs = env.reset()
        #print('-----------------------------')
        #print('Episode %d:'%(episode))
        print('reset observation:')
        #print(obs["desired_goal"])
        print(obs)
        print('-----------------------------')
        for _ in range(20):  # 10 seconds
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            #env.render()
            
            print('-----------------------------')
            print('action: %s'%(action))
            #print('state: %s, %s, %s'%(obs["observation"].shape, obs["desired_goal"].shape, obs["achieved_goal"].shape))
            print(obs)
            print('reward: %f'%(reward))
            print('succeed: '+str(info["is_success"]))
            print('info: ' + str(info))  # all of these envs have info['is_success']
            print('-------------------------------')
            '''
            '''
            step += 1
            if done:
                break
                    
        print('Episode finished after {} timesteps.'.format(step))

    print('-----------------------------')
    print("Action space: %s"%(env.action_space)) 
    print("Observation space: %s"%(env.observation_space))  
    print("Max episode steps: "+str(env.spec.max_episode_steps))
    print('-----------------------------') 
    
    env.close() 

if __name__ == "__main__":  
    # ---------------- gym ---------------
    # 'LunarLander-v2'  # action: 4(d), state: 8(c), max_len: 1000
    # 'InvertedDoublePendulum-v2' # action: 1(c), state: 11(c)
    # 'Swimmer-v2' # action: 2(c), state: 8(c), max_len: 1000
    # ---------------- d4rl ---------------
    # 'minigrid-fourrooms-v0' # action: 7(d), state: 7*7*3(c)
    # 'maze2d-umaze-v1' # action: 2(c), state: 4(c)
    # 'antmaze-umaze-v0' # action: 8(c), state: 29(c)
    # 'kitchen-complete-v0' # action: 9(c), state: 60(c)
    # ---------------- gcsl ---------------
    # 'Pointmass-Rooms' # action: 2(c), state: 2(c) = goal: 2(c)
    # 'SawyerPush' # action: 2(c), state: 4(c) = goal: 4(c)
    # 'SawyerDoor' # action: 3(c), state: 4(c) != goal: 1(c)
    # 'Claw' # action: 9(c), state: 11(c) != goal: 2(c)
    # 'Lunar' # action: 4(d), state: 8(c) != goal: 5(c)
    # ---------------- gym goal reaching -----------
    # 'FetchPickAndPlace-v1' # action: 2(c), state: 2(c) = goal: 2(c), fixed goal
    # 'FetchPush-v1' , fixed goal
    # 'FetchReach-v1', fixed goal
    # 'FetchSlide-v1', fixed goal
    
    #test_env(env_type='gym', env_id='Swimmer-v2')
    #test_env(env_type='gcsl', env_id='Lunar', fixed_start=False, fixed_goal=False)
    test_gym_env()
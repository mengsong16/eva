from rlkit.samplers.rollout_functions import rollout
from rlkit.samplers.rollout_functions import multitask_rollout
#from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv
import argparse
import torch
import uuid
from rlkit.core import logger
import os
from eva.utils.path import *

filename = str(uuid.uuid4())

def load_env_policy(model_file, gpu, policy_env_type, mode='video_env'):
    model_file = os.path.join(model_file, "params.pkl")
    data = torch.load(model_file)
    policy = data['%s/policy'%(policy_env_type)]
    if hasattr(policy, 'policy'):
        policy = policy.policy
    env = data['%s/env'%(policy_env_type)]
    print("Policy and environment loaded")

    if gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)

    if isinstance(env, VAEWrappedEnv) and hasattr(env, 'mode'):
        env.mode(mode)

    return env, policy    

def visualize_policy(model_file, goal_conditioned, policy_env_type, max_path_length=300, gpu=True):
    assert policy_env_type == "evaluation" or policy_env_type == "exploration", "Error: undefined policy type."
    env, policy = load_env_policy(model_file, gpu, policy_env_type)

    
    # some environments need to be reconfigured for visualization
    if hasattr(env, 'enable_render'):
        env.enable_render()
    
    while True:
        # rollout until max_path_length or done
        if goal_conditioned:
            # env can have random or fixed goal, the key is the goal is provided
            multitask_rollout(
                env,
                policy,
                max_path_length=max_path_length,
                render=True,
                return_dict_obs=True,
                observation_key='observation',
                desired_goal_key='desired_goal',
            )
        else:    
            rollout(
                env,
                policy,
                max_path_length=max_path_length,
                render=True,
            )

def evaluate_policy(model_file, goal_conditioned, num_eval_episodes, policy_env_type, gpu=True):
    assert policy_env_type == "evaluation" or policy_env_type == "exploration", "Error: undefined policy type."
    env, policy = load_env_policy(model_file, gpu, policy_env_type)

    # some environments need to be reconfigured for visualization
    if hasattr(env, 'enable_render'):
        env.enable_render()
    
    paths = []
    returns = []
    for i in range(num_eval_episodes):
        # rollout until max_path_length or done
        if goal_conditioned:
            # env can have random or fixed goal, the key is the goal is provided
            path = multitask_rollout(
                env,
                policy,
                render=False,
                return_dict_obs=True,
                observation_key='observation',
                desired_goal_key='desired_goal',
            )
        else:    
            path = rollout(
                env,
                policy,
                render=False,
            )

        paths.append(path)
        rewards = np.array(path["rewards"], dtype=np.float32)
        episode_return = np.sum(rewards)
        returns.append(episode_return)
        print('-----------------------------')
        print('Episode: %d'%(i))
        print('Steps: %d'%(rewards.shape[0]))
        print("Return: %f"%(episode_return))
        print('-----------------------------')       

    returns = np.array(returns, dtype=np.float32)
    print("-------------- Evaluation summary --------------")
    print("Number of episodes: %d"%(num_eval_episodes))
    print("Min return: %f"%(np.min(returns, axis=0)))
    print("Mean return: %f"%(np.mean(returns, axis=0)))
    print("Max return: %f"%(np.max(returns, axis=0)))
    print("Std of return: %f"%(np.std(returns, axis=0)))
    print("------------------------------------------------")

if __name__ == "__main__":
    model_file = os.path.join(checkpoints_path, 
            "dqn-cartpole-v0", 
            "dqn-cartpole-v0_2022_04_25_23_15_35_0000--s-2")

    # model_file = os.path.join(checkpoints_path, 
    #         "her-dqn-goalgridworld-v0", 
    #         "her-dqn-goalgridworld-v0_2022_04_26_00_58_08_0000--s-1")        
            
    # visualize_policy(model_file=model_file, 
    #                 goal_conditioned=False,
    #                 policy_env_type="evaluation")
    evaluate_policy(model_file=model_file, 
                    goal_conditioned=False, 
                    policy_env_type="evaluation", # exploration
                    num_eval_episodes=100)
from eva.utils.data_utils import parse_config, get_device
from eva.utils.path import *
from eva.algorithms.common import *
from eva.envs.common import *

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os
import numpy as np
from eva.models.state_goal import CatAbsoluteGoalState, CatRelativeGoalState 

class PPOTrainer:
    def __init__(self, config_filename="ppo.yaml"):
        self.config = yaml2variant(config_filename=config_filename)
        self.env_id = self.config.get("env_id")
        
        self.exp_prefix = ("%s-%s"%(self.config.get("algorithm"), self.env_id)).lower()
        self.exp_name = self.config.get("experiment_name")

        self.goal_format = self.config.get("goal_format")
        assert self.goal_format in ["absolute", "relative"], "Error: undefined goal format: %s"%(self.goal_format)

        self.device = torch.device('cuda:%d'%(self.config.get("gpu_id")))

        # create a single env
        self.env = create_env(self.env_id)

        # set everything
        self.seed = self.config.get("seed")
        seed_env(self.env, self.seed)
        seed_other(self.seed)
        
    def train(self):
        if self.goal_format == "absolute":
            encoder_class = CatAbsoluteGoalState
        elif self.goal_format == "relative":
            encoder_class = CatRelativeGoalState

        policy_kwargs = dict(
            features_extractor_class=encoder_class,
            features_extractor_kwargs=dict([('output_dim', int(self.config.get("output_dim"))), 
                ('hidden_dim', int(self.config.get("hidden_dim"))),
                ('hidden_layer', int(self.config.get("hidden_layer")))]))        

        # Parallel environments
        # env is a string or gym.Env
        env_vec = make_vec_env(self.env_id, 
                n_envs=int(self.config.get("n_envs")), 
                wrapper_class=get_wrapper_class(self.env))

        # create policy
        tensorboard_folder = os.path.join(tensorboard_path, self.exp_prefix)
        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)

        model = PPO('MultiInputPolicy', env_vec,
                learning_rate=float(self.config.get("learning_rate")),
                gamma=float(self.config.get("gamma")),
                n_steps=int(self.config.get("n_steps")), 
                batch_size=int(self.config.get("batch_size")), 
                n_epochs=int(self.config.get("n_epochs")), 
                gae_lambda=float(self.config.get("gae_lambda")), 
                clip_range=float(self.config.get("clip_range")),
                ent_coef=float(self.config.get("ent_coef")),
                device=self.device,
                seed=self.seed,
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_folder)

        # train
        model.learn(total_timesteps=int(self.config.get("total_timesteps")), 
            tb_log_name=self.exp_name)
        
        # save model
        checkpoints_folder = os.path.join(checkpoints_path, self.exp_prefix)
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        
        model.save(os.path.join(checkpoints_folder, self.exp_name))

    # return True or False
    def check_is_success(self, info):
        if is_instance_gym_goal_env(self.env):
            return info["is_success"]
        elif is_instance_gcsl_env(self.env):    
            return self.env.is_success()
        else:
            print("Error: must be either a gym goal env or a gcsl env")
            exit()    

    def eval(self, render=False): 
        # load model   
        checkpoints_folder = os.path.join(checkpoints_path, self.exp_prefix)
        model = PPO.load(os.path.join(checkpoints_folder, self.exp_name))
        print("Model loaded")

        num_test_episodes = int(self.config.get("num_test_episodes"))

        success_array = []
        for i in range(num_test_episodes):
            # run one episode
            obs = self.env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                if render:
                    self.env.render()

                if done:
                    success_array.append(float(self.check_is_success(info)))
                    break

        # print success rate    
        success_array = np.array(success_array, dtype=np.float32)
        success_rate = np.mean(success_array, axis=0)
        print("Success rate: %f"%(success_rate))

if __name__ == "__main__": 
    ppo_trainer = PPOTrainer()
    ppo_trainer.train()
    ppo_trainer.eval(render=True)
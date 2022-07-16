from eva.utils.data_utils import parse_config, get_device
from eva.utils.path import *
from eva.algorithms.common import *
from eva.envs.common import *

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os
import numpy as np
from eva.models.state_goal import CatAbsoluteGoalState, CatRelativeGoalState 
import datetime
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common import base_class  # pytype: disable=pyi-error
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


class PPOEvaluator:
    def __init__(self, config_filename="ppo.yaml"):
        self.config = yaml2variant(config_filename=config_filename)
        self.env_id = self.config.get("env_id")

        self.eval_env = create_env(self.env_id)

        self.goal_format = self.config.get("goal_format")
        assert self.goal_format in ["absolute", "relative"], "Error: undefined goal format: %s"%(self.goal_format)


        # get seed
        self.seed = self.config.get("seed")
        # seed evaluation env
        seed_env(self.eval_env, self.seed)
        # set everything
        seed_other(self.seed)

        # get experiment name
        self.exp_prefix = ("%s-%s-%s"%(self.config.get("algorithm"), self.env_id, self.goal_format)).lower()
    
    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = float(info.get("is_success"))
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def eval(self, render=False, verbose=True): 
        # load model   
        checkpoints_folder = os.path.join(checkpoints_path, self.exp_prefix)
        checkpoint_path = os.path.join(checkpoints_folder, self.config.get("eval_exp_name"))
        model = PPO.load(checkpoint_path)  # device='auto'
        print("====> Model loaded from %s"%(checkpoint_path))

        # Reset success rate buffer
        self._is_success_buffer = []

        eval_num_episodes = int(self.config.get("num_test_episodes"))

        # evaluate model for num_test_episodes
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            self.eval_env,
            n_eval_episodes=eval_num_episodes,
            render=render,
            deterministic=False,
            return_episode_rewards=True,
            warn=True,
            callback=self._log_success_callback,
        )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        success_rate = np.mean(np.array(self._is_success_buffer))

        if verbose:
            print("==========================================")
            print(f"Eval num episodes: {eval_num_episodes}")
            print(f"Episode reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            print(f"Success rate: {100 * success_rate:.2f}%")
            print("==========================================")
            

if __name__ == "__main__": 
    ppo_evaluator = PPOEvaluator()
    ppo_evaluator.eval()
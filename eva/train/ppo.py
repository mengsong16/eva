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

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 10,
        eval_freq: int = 100,
        deterministic: bool = False,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        #self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")


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

    def _on_step(self) -> bool:

        continue_training = True

        # should evaluate
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            # append success buffer with the success results of n_eval_episodes
            # if len(self._is_success_buffer) > 0:
            #     self.evaluations_successes.extend(self._is_success_buffer)
                   
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            
            #print("=================================")
            #print(episode_rewards)

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            
            # Add mean reward and mean ep length to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Add success rate of n_eval_episodes to current Logger
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(np.array(self._is_success_buffer))
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            # self.num_timesteps: steps collected so far = n_envs * n times env.step() was called
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class PPOTrainer:
    def __init__(self, config_filename="ppo.yaml"):
        self.config = yaml2variant(config_filename=config_filename)
        self.env_id = self.config.get("env_id")


        self.goal_format = self.config.get("goal_format")
        assert self.goal_format in ["absolute", "relative"], "Error: undefined goal format: %s"%(self.goal_format)

        self.device = torch.device('cuda:%d'%(self.config.get("gpu_id")))

        # create a single env for evaluation
        # will be used to get wrapper for vectorized envs
        self.eval_env = create_env(self.env_id)

        # get seed
        self.seed = self.config.get("seed")
        # seed evaluation env
        seed_env(self.eval_env, self.seed)
        # set everything
        seed_other(self.seed)

        # get experiment name
        self.exp_prefix = ("%s-%s-%s"%(self.config.get("algorithm"), self.env_id, self.goal_format)).lower()
        # experiment_name - YearMonthDay-HourMiniteSecond
        now = datetime.datetime.now()
        self.exp_name = "s%d-"%(self.seed)+now.strftime("%Y%m%d-%H%M%S").lower() 

        
    def train(self):
        if self.goal_format == "absolute":
            encoder_class = CatAbsoluteGoalState
        elif self.goal_format == "relative":
            encoder_class = CatRelativeGoalState
        else:
            print("Error: undefined goal format: %s"%(self.goal_format))
            exit()    

        policy_kwargs = dict(
            features_extractor_class=encoder_class,
            features_extractor_kwargs=dict([('output_dim', int(self.config.get("output_dim"))), 
                ('hidden_dim', int(self.config.get("hidden_dim"))),
                ('hidden_layer', int(self.config.get("hidden_layer")))]))        

        # Parallel environments
        # env is a string or gym.Env
        # make_vec_env return wrapped envs
        env_vec = make_vec_env(env_id=self.env_id, 
                n_envs=int(self.config.get("n_envs")), seed=self.seed,
                wrapper_class=get_wrapper_class(self.eval_env))

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

        # Use deterministic actions for evaluation
        eval_callback = EvalCallback(eval_env=self.eval_env,
                                    n_eval_episodes=10, 
                                    eval_freq=100, 
                                    deterministic=False, render=False)

        # train
        model.learn(total_timesteps=int(self.config.get("total_timesteps")), 
            tb_log_name=self.exp_name, callback=eval_callback)
        
        # save model
        checkpoints_folder = os.path.join(checkpoints_path, self.exp_prefix)
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        
        model.save(os.path.join(checkpoints_folder, self.exp_name))    

    

if __name__ == "__main__": 
    ppo_trainer = PPOTrainer()
    ppo_trainer.train()
    
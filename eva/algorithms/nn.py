import numpy as np
import torch
import gym

def seed_env(env: gym.Env, seed: int) -> None:
    """Set the random seed of the environment."""
    if seed is None:
        seed = np.random.randint(2 ** 31 - 1)
        
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def compute_loss(self, observations, expert_actions, loss_type='log_prob'):
    """Compute loss of self._learner on the expert_actions.
    """
    learner_output = self.learner(observations)
    if loss_type == 'mse':
        assert not(self.deterministic==True and self.discrete_action==True), "Error: policy must be a distribution for discrete action space"
        
        if isinstance(learner_output, torch.Tensor):
            # We must have a deterministic policy as the learner.
            learner_actions = learner_output
        else:
            # We must have a stochastic policy as the learner.
            action_dist, _ = learner_output
            learner_actions = action_dist.rsample()
        return torch.mean((expert_actions - learner_actions)**2)
    else:
        assert loss_type == 'log_prob'
        # We already checked that we have a StochasticPolicy as the learner
        action_dist, _ = learner_output
        return -torch.mean(action_dist.log_prob(expert_actions))
import numpy as np
import torch
import torch.nn as nn
from gym import spaces

class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        # Squash correction (from original SAC implementation)
        return torch.log(1.0 - torch.tanh(x) ** 2 + self.epsilon)

class Policy(nn.Module):

    """ Policy network takes state and target commands as input
	and returns probability distribution over all actions.
	"""

    # the same hidden size is used for all layers
    def __init__(self, state_dim, action_space, command_dim=0, deterministic=False, hidden_size=1024, hidden_layer=2, dropout=0, epsilon: float = 1e-6):
        super(Policy, self).__init__()

        if isinstance(action_space, spaces.Box):
            self.act_dim = action_space.n
            self.discrete_action = True
        else:
            self.act_dim = int(np.prod(action_space.shape))
            self.discrete_action = False

        self.deterministic = deterministic

        self.action_center = torch.tensor((action_space.high + action_space.low) / 2)
        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2)

        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon

        # mlp module
        assert hidden_layer >= 1, "Error: Must have at least one hidden layers"
        # hidden layer 1 (input --> hidden): linear+relu+dropout
        self.mlp_module = [
            nn.Linear(state_dim+command_dim, hidden_size),
            nn.ReLU()]

        if dropout > 0:    
            self.mlp_module.append(nn.Dropout(dropout))
        
        # hidden layer 2 to n (hidden --> hidden): linear+relu+dropout
        for _ in range(hidden_layer-1):
            self.mlp_module.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

            if dropout > 0:    
                self.mlp_module.append(nn.Dropout(dropout))
        
        # output head
        # discrete distribution
        if self.discrete_action == True:
            self.action_head = nn.Linear(hidden_size, self.act_dim)
        # continuous distribution
        else:
            self.action_mean_head = nn.Linear(hidden_size, self.act_dim)
            self.action_logstd_head = nn.Linear(hidden_size, self.act_dim)

        
    # features: [batch_size, state_dim + command_dim]
    # return [batch_size, 1] distributions
    def get_distributions(self, features):
        mlp_output = self._mlp_module(features)

        # discrete distribution
        if self.discrete_action == True:
            probs = torch.softmax(self.action_head(mlp_output), axis=1)
            return torch.distributions.Categorical(probs=probs)
        # continuous distribution
        else: 
            mean = self.action_mean_head(mlp_output)
            logstd = self.action_logstd_head(mlp_output).clamp(-20, 2)
            std = torch.ones_like(mean) * logstd.exp()
            return torch.distributions.Normal(mean, std)
    
    def squash_actions(self, actions):
        # squash to [-1,1]
        tanh_actions = torch.tanh(actions)
        return tanh_actions * self.action_scale + self.action_center

    def get_modes(self, features):
        dists = self.get_distributions(features)
        # discrete distribution
        if self.discrete_action == True:
            return torch.argmax(dists.probs, dim=1)
        # continuous distribution
        else:
            return self.squash_actions(dists.mean)
    
    # actions: [batch_size, action_dim] numpy array
    def sample_actions(self, features):
        dists = self.get_distributions(features) 
        # Reparametrization trick to pass gradients
        actions = dists.rsample()
        actions = self.squash_actions(actions)
        return actions.cpu().numpy()
    
    def sum_independent_dims(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Continuous actions are usually considered to be independent,
        so we can sum components of the ``log_prob`` or the entropy.

        :param tensor: shape: (n_batch, n_actions) or (n_batch,)
        :return: shape: (n_batch,)
        """
        if len(tensor.shape) > 1:
            tensor = tensor.sum(dim=1)
        else:
            tensor = tensor.sum()
        return tensor

    def log_prob(self, distribution, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        """

        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        # actions will be clipped to avoid NaN when inversing tanh
        actions = TanhBijector.inverse(actions)

        log_prob = distribution.log_prob(actions)
        log_prob = self.sum_independent_dims(log_prob)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - actions**2 + self.epsilon), dim=1)

        return log_prob
    
    # for training   
    # features: [batch_size, state_dim + command_dim]
    # return distributions or actions 
    def forward(self, features):
        if self.deterministic:
            return self.get_modes(features)
        else:
            return self.get_distributions(features)    
    
    # for evaluation
    # get actions from conditioned states
    # features: [batch_size, state_dim + command_dim]
    # actions: [batch_size, action_dim] numpy array
    def get_actions(self, features):
        with torch.no_grad():
            if self.deterministic:
                return self.get_modes(features)
            else:    
                return self.sample_actions(features)
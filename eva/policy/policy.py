import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from eva.policy.tanh_normal import TanhNormal


# for supervise learning
class Policy(nn.Module):

    """ Policy network takes state and target commands as input
	and returns probability distribution over all actions.
	"""

    # the same hidden size is used for all layers
    def __init__(self, state_dim, action_space, command_dim=0, deterministic=False, hidden_size=1024, hidden_layer=2, dropout=0):
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

    # for continunous action, scale actions from [-1,1] to [low, high] of the action space
    def scale_actions(self, actions):
        return actions * self.action_scale + self.action_center

    # features: [batch_size, state_dim + command_dim]
    # return [batch_size, 1] distributions
    # allow backprop
    def get_distributions(self, features):
        mlp_output = self._mlp_module(features)

        # discrete distribution
        if self.discrete_action == True:
            probs = torch.softmax(self.action_head(mlp_output), axis=1)
            return torch.distributions.Categorical(probs=probs)
        # continuous distribution
        else: 
            mean = self.action_mean_head(mlp_output)
            # clip logstd to [-20,2]
            logstd = self.action_logstd_head(mlp_output).clamp(-20, 2)
            std = torch.ones_like(mean) * logstd.exp()
            return TanhNormal(mean, std)
    
    # actions: [batch_size, action_dim] numpy array
    # allow backprop
    def sample_actions(self, features):
        dists = self.get_distributions(features) 
        # Use reparametrization trick to pass gradients
        actions = dists.rsample()
        
        return actions

    # allow backprop
    def get_deterministic_actions(self, features):
        mlp_output = self._mlp_module(features)

        # discrete action
        if self.discrete_action == True:
            probs = torch.softmax(self.action_head(mlp_output), axis=1)
            return torch.argmax(probs, dim=1)
        # continuous action
        else:
            mean = self.action_mean_head(mlp_output)
            return torch.tanh(mean)
    
    
    # for training   
    # features: [batch_size, state_dim + command_dim]
    # return actions: [batch_size, action_dim] tensor 
    def forward(self, features):
        if self.deterministic:
            return self.get_deterministic_actions(features)
        else:
            return self.sample_actions(features)    
    
    # for evaluation
    # features: [batch_size, state_dim + command_dim]
    # return actions: [batch_size, action_dim] numpy array
    def get_actions(self, features):
        with torch.no_grad():
            actions = self.forward(features)
        
            return actions.cpu().numpy()
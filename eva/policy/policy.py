import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from eva.policy.tanh_normal import TanhNormal

class RandomPolicy():
    def __init__(self, env):
        self.env = env
        if isinstance(self.env.action_space, spaces.Box): # continuous action space
            self.discrete_action = False
        else:
            self.discrete_action = True


    # return a single action
    def get_actions(self, aug_states):
        # uniform random policy
        # discrete: action is an interger
        # continuous: action is [1, action_dim]
        action = self.env.action_space.sample()
        if self.discrete_action == False:
            action = np.expand_dims(action, axis=0)

        return action   

# for supervise learning
class Policy(nn.Module):

    """ Policy network takes state and target targets as input
	and returns probability distribution over all actions.
	"""

    # the same hidden size is used for all layers
    def __init__(self, state_dim, action_space, target_dim=0, deterministic=False, hidden_size=1024, hidden_layer=2, dropout=0):
        super(Policy, self).__init__()

        if isinstance(action_space, spaces.Box): # continuous action space
            self.act_dim = int(np.prod(action_space.shape))
            self.discrete_action = False
        else:
            self.act_dim = action_space.n
            self.discrete_action = True

        self.deterministic = deterministic

        if self.discrete_action == False:
            self.action_center = torch.tensor((action_space.high + action_space.low) / 2)
            self.action_scale = torch.tensor((action_space.high - action_space.low) / 2)


        # mlp module
        assert hidden_layer >= 1, "Error: Must have at least one hidden layers"
        # hidden layer 1 (input --> hidden): linear+relu+dropout
        self.mlp_module = [
            nn.Linear(state_dim+target_dim, hidden_size),
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
            self.softmax = nn.Softmax(dim=1)
        # continuous distribution
        else:
            self.action_mean_head = nn.Linear(hidden_size, self.act_dim)
            self.action_logstd_head = nn.Linear(hidden_size, self.act_dim)

        self.mlp_module = nn.Sequential(*self.mlp_module)

    # for continunous action, scale actions from [-1,1] to [low, high] of the action space
    def scale_actions(self, actions):
        return actions * self.action_scale + self.action_center

    def get_device(self):
        return next(self.parameters()).device

    def from_numpy_to_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
            x = x.to(self.get_device())
        return x
    
    # aug_states: [batch_size, state_dim + target_dim]
    # return [batch_size, 1] distributions
    # allow backprop
    def get_distributions(self, aug_states):
        # ensure aug_states are tensors
        aug_states = self.from_numpy_to_tensor(aug_states)

        mlp_output = self.mlp_module(aug_states)

        # discrete distribution
        if self.discrete_action == True:
            probs = self.softmax(self.action_head(mlp_output))
            return torch.distributions.Categorical(probs=probs)
        # continuous distribution
        else: 
            mean = self.action_mean_head(mlp_output)
            # clip logstd to [-20,2]
            logstd = self.action_logstd_head(mlp_output).clamp(-20, 2)
            std = torch.ones_like(mean) * logstd.exp()
            return TanhNormal(mean, std)
    
    # actions: discrete: [batch_size], continuous: [batch_size, action_dim]
    # allow backprop
    def sample_actions(self, aug_states):
        dists = self.get_distributions(aug_states) 
        
        # Use reparametrization trick to pass gradients
        try:
            actions = dists.rsample()
        except NotImplementedError:
            # Categorical distribution does not implement rsample()
            # the output of sample() is not differentiable, thus requires_grad = Fasle
            actions = dists.sample()    
        
        return actions

    # allow backprop
    def get_deterministic_actions(self, aug_states):
        # ensure aug_states are tensors
        aug_states = self.from_numpy_to_tensor(aug_states)

        mlp_output = self.mlp_module(aug_states)

        # discrete action
        # return [B]
        if self.discrete_action == True:
            probs = self.softmax(self.action_head(mlp_output))
            # argmax is not differentiable, so its output's requires_grad = False
            return torch.argmax(probs, dim=1)
        # continuous action
        # return [B, action_dim]
        else:
            mean = self.action_mean_head(mlp_output)
            
            return torch.tanh(mean)
    
    
    # for training   
    # aug_states: [batch_size, state_dim + target_dim]
    # return actions: [batch_size, action_dim] tensor 
    def forward(self, aug_states):

        if self.deterministic:
            return self.get_deterministic_actions(aug_states)
        else:
            return self.sample_actions(aug_states)    
    
    # for evaluation
    # aug_states: [batch_size, state_dim + target_dim]
    # return actions: [batch_size, action_dim] numpy array
    def get_actions(self, aug_states):
        with torch.no_grad():
            actions = self.forward(aug_states)
        
            # to pass to environment
            return actions.cpu().numpy()
    
    # get log probability of given actions
    # the probability distributions are calculated based on aug_states
    # aug_states: [B, state_dim]
    # given_actions: discrete: [B], continuous: [B, action_dim]
    def get_log_probs(self, aug_states, given_actions):
        # dists: [B]
        dists = self.get_distributions(aug_states)
        
        return dists.log_prob(given_actions)

# input and output of softmax have the same shape: [B,x]
# output of argmax: [B]
def test_softmax():
    m = nn.Softmax(dim=1)
    input = torch.randn(2, 3)
    output = m(input)
    print(output.size())
    print(torch.argmax(output, dim=1).size())

if __name__ == "__main__":
    test_softmax()
    print("Done.")
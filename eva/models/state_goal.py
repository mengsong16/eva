import numpy as np
import torch
import torch.nn as nn
from gym import spaces
import gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# assume state is a dictionary {observation, achieved_goal, desired_goal}
# parameters: observation_space: gym.Space, features_dim: int
class GoalConditionedState(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(GoalConditionedState, self).__init__(observation_space, features_dim)

        
    def create_mlp(self, input_dim, output_dim, hidden_dim, hidden_layer, dropout):
        # mlp module
        assert hidden_layer >= 1, "Error: Must have at least one hidden layers"
        # hidden layer 1 (input --> hidden): linear+relu+dropout
        self.mlp_module = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()]

        if dropout > 0:    
            self.mlp_module.append(nn.Dropout(dropout))
        
        # hidden layer 2 to n (hidden --> hidden): linear+relu+dropout
        for _ in range(hidden_layer-1):
            self.mlp_module.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])

            if dropout > 0:    
                self.mlp_module.append(nn.Dropout(dropout))
        
        # last layer n+1 (hidden --> output)
        self.mlp_module.extend([
            nn.Linear(hidden_dim, output_dim)
        ])

        self.mlp_module = nn.Sequential(*self.mlp_module)    
    
    def get_device(self):
        return next(self.parameters()).device

    def from_numpy_to_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.get_device())
        return x

    def forward(self, states):
        # ensure states are tensors on the same device
        states = self.from_numpy_to_tensor(states)

        return self.mlp_module(states)
    
    # for evaluation, return numpy arrays
    def eval(self, states):
        with torch.no_grad():
            output = self.forward(states)

            return output.cpu().numpy()

class CatAbsoluteGoalState(GoalConditionedState):
    def __init__(self, observation_space, output_dim, hidden_dim, hidden_layer=2, dropout=0):
        super(CatAbsoluteGoalState, self).__init__(observation_space=observation_space, features_dim=output_dim)

        input_dim = observation_space["observation"].shape[0] + observation_space["desired_goal"].shape[0]

        self.create_mlp(input_dim, output_dim, hidden_dim, hidden_layer, dropout)
    
    def forward(self, states):
        cat_states = torch.cat((states["observation"], states["desired_goal"]), dim=1)

        return super(CatAbsoluteGoalState, self).forward(cat_states)

class CatRelativeGoalState(GoalConditionedState):
    def __init__(self, observation_space, output_dim, hidden_dim, hidden_layer=2, dropout=0):
        super(CatRelativeGoalState, self).__init__(observation_space=observation_space, features_dim=output_dim)

        input_dim = observation_space["observation"].shape[0] + observation_space["desired_goal"].shape[0]
        
        self.create_mlp(input_dim, output_dim, hidden_dim, hidden_layer, dropout)
    
    def forward(self, states):
        cat_states = torch.cat((states["observation"], states["desired_goal"]-states["achieved_goal"]), dim=1)

        return super(CatAbsoluteGoalState, self).forward(cat_states)

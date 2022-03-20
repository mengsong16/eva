import numpy as np
import torch
import torch.nn as nn


class ReturnConditionedPolicy(nn.Module):

    """ Policy network takes state and target commands as input
	and returns probability distribution over all actions.
	"""

    # the same hidden size is used for all layers
    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, **kwargs):
        super(ReturnConditionedPolicy, self).__init__()


        # layer 0: linear
        layers = [nn.Linear(state_dim, hidden_size)]
        # layer 1 to n-1: relu+dropout+linear
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        # layer n: relu+dropout_linear+tanh    
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    # input: a sequence of states of length max_length
    # output: a sequence of actions of length max_length
    # for training
    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

    # input a sequence of states of length max_length
    # only return the last action
    # for evaluation
    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        # pad with 0 to states if shorter than max_length
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1]
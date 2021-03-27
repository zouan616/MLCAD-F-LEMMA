import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import namedtuple
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


actions = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(19, 64)
        # self.affine2 = nn.Linear(32,64)


        # actor's layer
        # self.action = nn.Linear(64,64)
        self.action_head = nn.Linear(64,len(actions))
        # self.action_head = nn.Linear(64, 2)

        # critic's layer
        self.value_head = nn.Linear(64, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # x = F.relu(self.affine2(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        # fst = F.relu(self.action(x))

        # mu, sigma = self.action_head(x)
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

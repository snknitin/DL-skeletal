import torch.nn as nn
import torch
from torch.nn.functional import  dropout
from torch.nn import Linear, LeakyReLU,ReLU, BatchNorm1d, ModuleList, L1Loss, LeakyReLU, Dropout, MSELoss,SELU

class DQN(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())





class DeepDQN(nn.Module):
    """
    Simple MLP network
    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128, middle_size: int = 64):
        super(DeepDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            LeakyReLU(),
            Dropout(p=0.2),
            nn.Linear(hidden_size, middle_size),
            LeakyReLU(),
            Dropout(p=0.2),
            nn.Linear(middle_size, n_actions)
        )

    def forward(self, x):
        return self.network(x.float())



import numpy as np

# class BranchDuelingDQN(nn.Module):
#     def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
#         super().__init__()
#         self.feature = nn.Sequential(
#             nn.Linear(obs_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
#         )
#         self.value = nn.Linear(hidden_size, 1)
#         self.advantage = nn.Linear(hidden_size, n_actions)
#
#     def forward(self, x):
#         x = x.float()
#         feature = self.feature(x)
#         value = self.value(feature)
#         advantage = self.advantage(feature)
#         return value + advantage - advantage.mean(dim=-1, keepdim=True)
#
#     def act(self, obs, epsilon: float = 0.0):
#         if np.random.random() < epsilon:
#             return np.random.randint(0, self.advantage.out_features)
#         else:
#             with torch.no_grad():
#                 q_values = self(obs.unsqueeze(0))
#                 return torch.argmax(q_values, dim=1).item()


import torch
import torch.nn as nn

class BranchDuelingDQN(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128,middle_size: int = 64,out_size: int = 32):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            # Dropout(p=0.2),
            nn.Linear(hidden_size, middle_size),
            nn.ReLU(),
            # Dropout(p=0.2),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            # Dropout(p=0.2),
            nn.Linear(middle_size, out_size),
            nn.ReLU(),
            # Dropout(p=0.2)
        )
        self.value = nn.Linear(out_size, 1)
        self.advantage = nn.Linear(out_size, n_actions)

    def forward(self, x):
        x = x.float()
        feature = self.feature(x)
        value = self.value(feature)
        advantage = self.advantage(feature)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class BranchDuelingDQNMulti(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, num_fcs: int, hidden_size: int = 128,middle_size: int = 64,out_size: int = 32):
        super().__init__()
        self.num_fcs = num_fcs
        self.n_actions = n_actions
        self.obs_size = obs_size * num_fcs
        self.feature = nn.Sequential(
            nn.Linear(self.obs_size, hidden_size),
            nn.ReLU(),
            # Dropout(p=0.2),
            nn.Linear(hidden_size, middle_size),
            nn.ReLU(),
            # Dropout(p=0.2),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            # Dropout(p=0.2),
            nn.Linear(middle_size, out_size),
            nn.ReLU(),
            # Dropout(p=0.2)
        )
        self.value = nn.Linear(out_size, self.num_fcs)
        self.advantage = nn.Linear(out_size, self.num_fcs * self.n_actions)

    def forward(self, x):
        x = x.float()
        feature = self.feature(x)
        value = self.value(feature).unsqueeze(-1)
        advantage = self.advantage(feature).view(-1, self.num_fcs, self.n_actions)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)